//! The router LP hands a stamped cancel on along its request's path, against a
//! scripted clock authority and two scripted workers. Requests a and b are
//! placed, then both clients hang up. Request a's cancel arrives before its
//! relay, so it goes to prefill on `http` and a's late prefill response is read
//! and dropped; b's arrives after its relay, so it goes to decode on `relay`,
//! and b's stream event is still read and sent on to traffic.

use std::{
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

/// A worker: it answers a cancel 204; prefill answers a request once the
/// gate lets it, with the relay stamp its prompt names; decode streams one
/// event and ends, as a decode hung up by a cancel does.
async fn worker(
    State((seen, prefill, gate)): State<(Seen, bool, Arc<Semaphore>)>,
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
    } else if prefill {
        gate.acquire().await.unwrap().forget();
        let stamp = match body["prompt"].as_str() {
            Some("a") => "a:1.5;s:0",
            _ => "a:1.625;s:1",
        };
        Json(json!({"kv_transfer_params": {"compass": stamp, "dp_rank": 0}})).into_response()
    } else {
        let events = ": compass a=2.5 s=0\ndata: x\n\n";
        ([("content-type", "text/event-stream")], events).into_response()
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
        let requests = json!({"kind": "GRANT", "G": 0.7, "released": {
            "traffic->router:http": [[0, 0.5], [1, 0.75]],
        }});
        write_frame(&mut sock, requests).await;
        let placed = read_frame(&mut sock).await;
        ready.await.unwrap();
        let rest = json!({"kind": "GRANT", "G": 3.0, "released": {
            "traffic->router:http": [[2, 1.0], [3, 1.75]],
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
    let gate = Arc::new(Semaphore::new(0));
    for prefill in [true, false] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/completions", post(worker))
            .route(
                "/kv_transfer_info",
                get(|| async { Json(json!({"tp_size": 1, "kv_role": "kv_producer"})) }),
            )
            .with_state((seen.clone(), prefill, gate.clone()));
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
        context.worker_registry.register(worker);
    }
    let router = RouterFactory::create_router(&context).await.unwrap();
    let state = Arc::new(AppState {
        router: Arc::from(router),
        context: context.clone(),
        router_manager: None,
    });
    let app = build_app(state, 1 << 20, vec![]);

    let clients: Vec<_> = [("a", "a:0.5;s:0"), ("b", "a:0.75;s:1")]
        .into_iter()
        .map(|(prompt, stamp)| {
            let body = json!({"model": "m", "prompt": prompt, "stream": true});
            tokio::spawn(app.clone().oneshot(request(stamp, body.to_string())))
        })
        .collect();
    let both_at_prefill = async {
        while seen.lock().unwrap().len() < 2 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), both_at_prefill)
        .await
        .expect("both requests are placed and posted to prefill");
    for client in clients {
        client.abort();
    }
    let cancels: Vec<_> = [("a:1.0;s:2", 0), ("a:1.75;s:3", 1)]
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
        ])
    );
    assert_eq!(
        handled["log"],
        json!([
            ["router->frontend-P:http", 2, 1.125],
            ["router->frontend-D:relay", 0, 2.125],
            ["router->frontend-D:relay", 1, 2.25],
            ["router->traffic:stream", 0, 2.75],
        ])
    );
    // Two requests and a cancel to prefill; b's request and b's cancel to decode.
    let all_posted = async {
        while seen.lock().unwrap().len() < 5 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), all_posted)
        .await
        .expect("each cancel is posted on");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 5, "{seen:?}");
    let cancel_at_prefill = (
        true,
        Some("compass=a:1.125;s:2".to_owned()),
        json!({"compass_cancel": 0}),
    );
    assert!(seen.contains(&cancel_at_prefill), "{seen:?}");
    let to_decode: Vec<_> = seen.iter().filter(|(p, ..)| !p).map(|s| &s.2).collect();
    assert!(
        to_decode.iter().any(|body| body["prompt"] == "b"),
        "{seen:?}"
    );
    let cancel_at_decode =
        json!({"compass_cancel": 0, "kv_transfer_params": {"compass": "a:2.25;s:1"}});
    assert!(to_decode.contains(&&cancel_at_decode), "{seen:?}");
}
