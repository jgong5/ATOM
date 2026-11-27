// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! The router LP ends the run when the ATOM PD router fails a request after
//! its relay went to decode, against a scripted clock authority and two
//! scripted workers. The router reads its prefill's TP size at its start; the
//! prefill the request is placed on registers after it, as a `--dp-aware`
//! rank can, so the router fails the request once the prefill has answered and
//! its relay is stamped. Decode would wait on the clock for that request, so
//! the router exits 1, naming it. `fail` ends the process, so the request runs
//! in a child process of this test.

use std::{process::Command, sync::Arc};

use axum::{
    body::Body,
    http::Request,
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
use serde_json::json;
use tokio::net::TcpListener;
use tower::ServiceExt;

mod compass_frames;
use compass_frames::{read_frame, write_frame};

const TEST: &str = "a_request_failed_after_its_relay_ends_the_run_naming_it";
const CHILD: &str = "COMPASS_RELAY_FAILURE_CHILD";

/// A worker serving `/v1/completions`: prefill answers with the relay stamp
/// its frontend gives, decode with one event.
async fn worker(prefill: bool) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let completions = move || async move {
        if prefill {
            Json(json!({"kv_transfer_params": {"compass": "a:1.5;s:0", "dp_rank": 0}}))
                .into_response()
        } else {
            let events = ": compass a=2.0 s=0\ndata: [DONE]\n\n";
            ([("content-type", "text/event-stream")], events).into_response()
        }
    };
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .route(
            "/kv_transfer_info",
            get(|| async { Json(json!({"tp_size": 1, "kv_role": "kv_producer"})) }),
        );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{address}")
}

fn registered(context: &AppContext, url: &str, kind: WorkerType) {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .model_id("m")
            .worker_type(kind)
            .build(),
    );
    context.worker_registry.register(worker);
}

/// The request, in the child: it returns only if the router answers it.
async fn request_on_a_prefill_registered_late() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("tcp://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let grant = json!({"kind": "GRANT", "G": 3.0, "released": {
            "traffic->router:http": [[0, 0.5]],
            "frontend-P->router:relay": [[0, 1.5]],
        }});
        for reply in [json!({"kind": "JOINED", "now": 0.0, "seqs": {}}), grant] {
            read_frame(&mut sock).await;
            write_frame(&mut sock, reply).await;
        }
        // Nothing more is released: the router's next ask waits here.
        read_frame(&mut sock).await;
        std::future::pending::<()>().await
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
        // As a simulated launch runs the router LP: a retry would place the
        // request again, and hold its relay again, past the run's clock.
        disable_retries: true,
        ..Default::default()
    };
    let prepare_runtime =
        mesh::core::prepare_pool::PreparePoolRuntime::new(config.resolved_prepare_pool().unwrap())
            .unwrap();
    let context = Arc::new(
        AppContext::from_config(config, 5, prepare_runtime.handle())
            .await
            .unwrap(),
    );
    let prefill = WorkerType::Prefill {
        bootstrap_port: None,
    };
    let (early, late, decode) = (worker(true).await, worker(true).await, worker(false).await);
    registered(&context, &early, prefill.clone());
    registered(&context, &decode, WorkerType::Decode);
    let router = RouterFactory::create_router(&context).await.unwrap();
    // Registered after the router read the prefills' TP sizes, it alone serves.
    context.worker_registry.remove_by_url(&early);
    registered(&context, &late, prefill);
    let state = Arc::new(AppState {
        router: Arc::from(router),
        context: context.clone(),
        router_manager: None,
    });
    let app = build_app(state, 1 << 20, vec![]);

    let request = Request::post("/v1/completions")
        .header("content-type", "application/json")
        .header("tracestate", "compass=a:0.5;s:0")
        .body(Body::from(
            json!({"model": "m", "prompt": "hi", "stream": true}).to_string(),
        ))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    eprintln!("answered {}", response.status());
}

#[test]
fn a_request_failed_after_its_relay_ends_the_run_naming_it() {
    if std::env::var_os(CHILD).is_some() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        return runtime.block_on(request_on_a_prefill_registered_late());
    }
    let child = Command::new(std::env::current_exe().unwrap())
        .args([TEST, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&child.stderr);
    assert_eq!(child.status.code(), Some(1), "{said}");
    let named = "router LP: request 0 failed 500 Internal Server Error after its relay \
                 went on router->frontend-D:relay as seq 0: ";
    let line = said.lines().find(|l| l.starts_with(named));
    let line = line.unwrap_or_else(|| panic!("no line naming the failure in {said}"));
    assert!(line.contains("tp_size missing for prefill"), "{line}");
    assert!(line.ends_with("; ending the run"), "{line}");
}
