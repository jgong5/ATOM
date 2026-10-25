//! The router LP hands a stamped cancel on along its request's path, against a
//! scripted clock authority and two scripted workers. Requests a, b and c are
//! placed, then their clients hang up. Request a's cancel arrives before its
//! relay, so it goes to prefill on `http` and a's late prefill response is read
//! and dropped; b's arrives after its relay, so it goes to decode on `relay`,
//! and b's stream event is still read and sent on to traffic. Each cancel drops
//! the loads its request holds in its own turn: c's prefill answers its cancel
//! 499 with no stamp, and b's stream stays open, until every message is handled.

use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use mesh::{
    app_context::AppContext,
    compass_clock::ENDPOINT_ENV,
    compass_router::{self, LOOKAHEADS_ENV, SEED_ENV},
    config::{types::BackendType, RouterConfig, RoutingMode},
    core::{BasicWorkerBuilder, Worker, WorkerType},
    routers::RouterFactory,
    server::{build_app, AppState},
};
use serde_json::{json, Value};
use tokio::{net::TcpListener, sync::Semaphore};
use tower::ServiceExt;

mod compass_frames;
use compass_frames::{read_frame, write_frame};

/// What each worker was sent: (prefill, tracestate, body).
type Seen = Arc<Mutex<Vec<(bool, Option<String>, Value)>>>;

/// A worker: it answers a cancel 204; prefill answers a and b once `gate`
/// lets it, with the relay stamp its prompt names, and c 499 with no stamp
/// once `later` lets it, as a prefill hung up by a cancel does; decode streams
/// one event and ends with no `[DONE]` once `later` lets it.
async fn worker(
    State((seen, prefill, gate, later)): State<(Seen, bool, Arc<Semaphore>, Arc<Semaphore>)>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    let state = headers
        .get("tracestate")
        .map(|h| h.to_str().unwrap().to_owned());
    let body: Value = serde_json::from_slice(&body).unwrap();
    seen.lock().unwrap().push((prefill, state, body.clone()));
    if body.get("compass_cancel").is_some() {
        StatusCode::NO_CONTENT.into_response()
    } else if prefill && body["prompt"] == "c" {
        later.acquire().await.unwrap().forget();
        let status = StatusCode::from_u16(499).unwrap();
        (status, Json(json!({"detail": "client disconnected"}))).into_response()
    } else if prefill {
        gate.acquire().await.unwrap().forget();
        let stamp = match body["prompt"].as_str() {
            Some("a") => "a:1.5;s:0",
            _ => "a:1.625;s:1",
        };
        Json(json!({"kv_transfer_params": {"compass": stamp, "dp_rank": 0}})).into_response()
    } else {
        let event = Ok::<_, Infallible>(Bytes::from(": compass a=2.5 s=0\ndata: x\n\n"));
        let end = futures_util::stream::once(async move {
            later.acquire().await.unwrap().forget();
            None
        });
        let events = futures_util::stream::iter([Some(event)]).chain(end);
        let body = Body::from_stream(events.filter_map(std::future::ready));
        ([("content-type", "text/event-stream")], body).into_response()
    }
}

fn request(stamp: &str, body: String) -> Request<Body> {
    Request::post("/v1/completions")
        .header("content-type", "application/json")
        .header("tracestate", format!("compass={stamp}"))
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancel_goes_to_prefill_before_the_relay_and_to_decode_after() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("tcp://{}", listener.local_addr().unwrap());
    let (go, ready) = tokio::sync::oneshot::channel::<()>();
    let served = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        read_frame(&mut sock).await;
        write_frame(&mut sock, json!({"kind": "JOINED", "now": 0.0, "seqs": {}})).await;
        read_frame(&mut sock).await;
        let requests = json!({"kind": "GRANT", "G": 0.9, "released": {
            "traffic->router:http": [[0, 0.5], [1, 0.75], [2, 0.875]],
        }});
        write_frame(&mut sock, requests).await;
        let placed = read_frame(&mut sock).await;
        ready.await.unwrap();
        let rest = json!({"kind": "GRANT", "G": 3.0, "released": {
            "traffic->router:http": [[3, 1.0], [4, 1.25], [5, 1.75]],
            "frontend-P->router:relay": [[0, 1.5], [1, 1.625]],
            "frontend-D->router:stream": [[0, 2.5]],
        }});
        write_frame(&mut sock, rest).await;
        let handled = read_frame(&mut sock).await;
        write_frame(
            &mut sock,
            json!({"kind": "GRANT", "G": "+inf", "released": {}}),
        )
        .await;
        (placed, handled)
    });
    std::env::set_var(ENDPOINT_ENV, &endpoint);
    std::env::set_var(SEED_ENV, "7");
    std::env::set_var(
        LOOKAHEADS_ENV,
        r#"{"router->traffic:stream": 0.25, "router->frontend-D:relay": 0.5,
            "router->frontend-P:http": 0.125}"#,
    );
    compass_router::start();

    let config = RouterConfig {
        backend: BackendType::Atom,
        mode: RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        },
        ..Default::default()
    };
    let context = Arc::new(AppContext::from_config(config, 5).await.unwrap());
    let seen = Seen::default();
    let (gate, later) = (Arc::new(Semaphore::new(0)), Arc::new(Semaphore::new(0)));
    let mut workers = Vec::new();
    for prefill in [true, false] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/completions", post(worker))
            .route(
                "/kv_transfer_info",
                get(|| async { Json(json!({"tp_size": 1, "kv_role": "kv_producer"})) }),
            )
            .with_state((seen.clone(), prefill, gate.clone(), later.clone()));
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let kind = if prefill {
            WorkerType::Prefill {
                bootstrap_port: None,
            }
        } else {
            WorkerType::Decode
        };
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("http://{address}"))
                .model_id("m")
                .worker_type(kind)
                .build(),
        );
        context.worker_registry.register(worker.clone());
        workers.push(worker);
    }
    let router = RouterFactory::create_router(&context).await.unwrap();
    let state = Arc::new(AppState {
        router: Arc::from(router),
        context: context.clone(),
        router_manager: None,
    });
    let app = build_app(state, 1 << 20, vec![]);

    let clients: Vec<_> = [
        ("a", "a:0.5;s:0"),
        ("b", "a:0.75;s:1"),
        ("c", "a:0.875;s:2"),
    ]
    .into_iter()
    .map(|(prompt, stamp)| {
        let body = json!({"model": "m", "prompt": prompt, "stream": true});
        tokio::spawn(app.clone().oneshot(request(stamp, body.to_string())))
    })
    .collect();
    let all_at_prefill = async {
        while seen.lock().unwrap().len() < 3 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), all_at_prefill)
        .await
        .expect("every request is placed and posted to prefill");
    for client in clients {
        client.abort();
    }
    let cancels: Vec<_> = [("a:1.0;s:3", 0), ("a:1.25;s:4", 2), ("a:1.75;s:5", 1)]
        .into_iter()
        .map(|(stamp, seq)| {
            let body = format!(r#"{{"compass_cancel": {seq}}}"#);
            tokio::spawn(app.clone().oneshot(request(stamp, body)))
        })
        .collect();
    gate.add_permits(2);
    go.send(()).unwrap();
    let (placed, handled) = tokio::time::timeout(Duration::from_secs(10), served)
        .await
        .expect("every released message is read though its client hung up")
        .unwrap();
    // c's prefill has not answered and b's stream has not ended.
    let loads: Vec<_> = workers.iter().map(|w| w.load()).collect();
    assert_eq!(loads, [0, 0], "each cancel drops its request's loads");
    later.add_permits(2);
    for cancel in cancels {
        assert_eq!(
            cancel.await.unwrap().unwrap().status(),
            StatusCode::NO_CONTENT
        );
    }

    assert_eq!(
        placed["log"],
        json!([
            ["router->frontend-P:http", 0, 0.625],
            ["router->frontend-P:http", 1, 0.875],
            ["router->frontend-P:http", 2, 1.0],
        ])
    );
    assert_eq!(
        handled["log"],
        json!([
            ["router->frontend-P:http", 3, 1.125],
            ["router->frontend-P:http", 4, 1.375],
            ["router->frontend-D:relay", 0, 2.125],
            ["router->frontend-D:relay", 1, 2.25],
            ["router->traffic:stream", 0, 2.75],
        ])
    );
    // Three requests and two cancels to prefill; b's request and b's cancel to decode.
    let all_posted = async {
        while seen.lock().unwrap().len() < 7 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), all_posted)
        .await
        .expect("each cancel is posted on");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 7, "{seen:?}");
    for (stamp, request) in [("a:1.125;s:3", 0), ("a:1.375;s:4", 2)] {
        let state = Some(format!("compass={stamp}"));
        let cancel_at_prefill = (true, state, json!({"compass_cancel": request}));
        assert!(seen.contains(&cancel_at_prefill), "{seen:?}");
    }
    let to_decode: Vec<_> = seen.iter().filter(|(p, ..)| !p).map(|s| &s.2).collect();
    assert!(
        to_decode.iter().any(|body| body["prompt"] == "b"),
        "{seen:?}"
    );
    let cancel_at_decode =
        json!({"compass_cancel": 0, "kv_transfer_params": {"compass": "a:2.25;s:1"}});
    assert!(to_decode.contains(&&cancel_at_decode), "{seen:?}");
}
