//! The router as logical process `router` of a simulated run.
//!
//! With `ATOM_COMPASS_CLOCK_ENDPOINT` set, the router binds LP `router` at
//! startup and handles the messages the clock releases to it one at a time, in
//! (arrival, channel, seq) order: each request from traffic, each prefill
//! response and each decode stream event. A message waits at its hold point
//! until it is released; its handler then runs with the policies' clock at its
//! arrival, and the next message is released only once it is handled. Each
//! message sends exactly one on, on the outbound channel of its kind (`http`,
//! `relay` or `stream`) to the frontend it goes to, stamped at its arrival plus
//! that channel's lookahead, from `ATOM_COMPASS_ROUTER_LOOKAHEADS`
//! (`{"<channel>": seconds}`); a request is sent on once it is placed. Where
//! several frontends take one kind, `ATOM_COMPASS_ROUTER_WORKERS`
//! (`{"<worker base url>": "<frontend LP>"}`) names each worker's. The
//! policies draw from the run's seed, `ATOM_COMPASS_SEED`. With
//! `ATOM_COMPASS_ROUTER_ORDER` (`[["<worker base url>", <DP rank>], ...]`) the
//! registry lists the workers in that order ([`determinism::set_worker_order`]).
//! A periodic task
//! fires on the LP clock instead of a thread ([`periodic`]): its next deadline
//! goes out as the daemon deadline of each request, and it runs before any
//! message arriving at or after that deadline. Any clock error ends the
//! process. Without the endpoint nothing here runs.
//!
//! A stamped cancel is answered 204 and goes on to where its request is
//! ([`ingress`]). A request is handled apart from its client's connection, so
//! a hang-up drops only the response; a prefill response read after its
//! request's cancel sends nothing on.

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::{stream::BoxStream, Stream, StreamExt};
use memchr::memmem;
use serde_json::Value;
use tokio::sync::{oneshot, Notify};
use tracing::{error, warn};

use crate::{
    compass_clock::{ClockClient, ClockError, ENDPOINT_ENV},
    core::WorkerLoadGuard,
    determinism,
};

pub const SEED_ENV: &str = "ATOM_COMPASS_SEED";
pub const LOOKAHEADS_ENV: &str = "ATOM_COMPASS_ROUTER_LOOKAHEADS";
pub const WORKERS_ENV: &str = "ATOM_COMPASS_ROUTER_WORKERS";
pub const ORDER_ENV: &str = "ATOM_COMPASS_ROUTER_ORDER";

/// A released message being handled. It holds the clock client, for the one
/// message it sends on; dropping it hands the client back and marks it handled.
pub struct Turn {
    arrival: f64,
    client: Option<ClockClient>,
    back: Option<oneshot::Sender<ClockClient>>,
    /// A prefill response read after its request's cancel, which goes no further.
    pub cancelled: bool,
}

impl Turn {
    /// Sends this message on, on `channel`, at its arrival plus that channel's
    /// lookahead; the stamp it goes with.
    fn send(&mut self, channel: &str) -> (f64, u64) {
        let lookahead = LP.get().and_then(|lp| lp.lookaheads.get(channel));
        let Some(lookahead) = lookahead else {
            fail(format!("no outbound channel {channel}"));
        };
        let arrival = self.arrival + lookahead;
        let client = self.client.as_mut().expect("a turn holds the clock");
        (arrival, client.send(channel, arrival))
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        if let (Some(client), Some(back)) = (self.client.take(), self.back.take()) {
            let _ = back.send(client);
        }
    }
}

struct Lp {
    /// Outbound channel -> lookahead.
    lookaheads: HashMap<String, f64>,
    /// Worker base URL -> its frontend LP.
    workers: BTreeMap<String, String>,
    /// (channel, seq) of each message read and not yet released.
    waiting: parking_lot::Mutex<HashMap<(String, u64), oneshot::Sender<Turn>>>,
    arrived: Notify,
    /// Periodic tasks, in registration order.
    timers: parking_lot::Mutex<Vec<Timer>>,
    /// Each request placed and not yet ended, by its seq on `traffic->router:http`.
    requests: parking_lot::Mutex<HashMap<u64, Placed>>,
    client: reqwest::Client,
}

/// Where a request is: the outbound channel it last went on, the worker's
/// base URL and its seq there, whether a cancel has come for it, and its
/// decode's base URL until it goes there.
#[derive(Default)]
struct Placed {
    channel: String,
    url: String,
    seq: u64,
    cancelled: bool,
    decode: String,
}

struct Timer {
    due: f64,
    every: f64,
    stopped: Arc<AtomicBool>,
    task: Box<dyn Fn() + Send>,
}

static LP: OnceLock<Lp> = OnceLock::new();

/// The request being handled: its seq on `traffic->router:http`, its turn
/// until it is placed, then its prefill stamp and its (prefill, decode)
/// frontends.
#[derive(Default)]
struct Handling {
    seq: u64,
    turn: Option<Turn>,
    stamp: Option<String>,
    pair: Option<(String, String)>,
}

tokio::task_local! {
    static REQUEST: RefCell<Handling>;
}

pub(crate) fn fail(why: impl std::fmt::Display) -> ! {
    error!("router LP: {why}; ending the run");
    eprintln!("router LP: {why}; ending the run");
    std::process::exit(1)
}

/// Bind LP `router` when the clock endpoint is set; call it before the
/// policies are built.
pub fn start() {
    let Ok(endpoint) = std::env::var(ENDPOINT_ENV) else {
        return;
    };
    let seed = std::env::var(SEED_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| fail(format!("{SEED_ENV} is not an unsigned integer")));
    let lookaheads = std::env::var(LOOKAHEADS_ENV)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| fail(format!("{LOOKAHEADS_ENV} is not a channel map")));
    let workers = std::env::var(WORKERS_ENV)
        .map_or_else(|_| Some(BTreeMap::new()), |s| serde_json::from_str(&s).ok());
    let workers = workers.unwrap_or_else(|| fail(format!("{WORKERS_ENV} is not a worker map")));
    let order = std::env::var(ORDER_ENV)
        .map_or_else(|_| Some(Vec::new()), |s| serde_json::from_str(&s).ok());
    let order = order.unwrap_or_else(|| fail(format!("{ORDER_ENV} is not a list of [url, rank]")));
    determinism::enable(seed);
    determinism::set_worker_order(order);
    let lp = LP.get_or_init(|| Lp::new(lookaheads, workers));
    tokio::spawn(async move {
        let result = async {
            let client = ClockClient::bind(&endpoint, "router").await?;
            lp.serve(client).await
        };
        if let Err(error) = result.await {
            fail(error);
        }
    });
}

fn kind(channel: &str) -> &str {
    channel.rsplit(':').next().unwrap_or_default()
}

fn entry((arrival, seq): (f64, u64)) -> String {
    format!("a:{arrival:?};s:{seq}")
}

impl Lp {
    fn new(lookaheads: HashMap<String, f64>, workers: BTreeMap<String, String>) -> Self {
        Self {
            lookaheads,
            workers,
            waiting: Default::default(),
            arrived: Notify::new(),
            timers: Default::default(),
            requests: Default::default(),
            client: reqwest::Client::new(),
        }
    }

    /// The earliest deadline of a task still running.
    fn next_due(&self) -> f64 {
        let mut timers = self.timers.lock();
        timers.retain(|t| !t.stopped.load(Ordering::Relaxed));
        timers.iter().map(|t| t.due).fold(f64::INFINITY, f64::min)
    }

    /// Run each task due by `time`, in deadline then registration order, with
    /// the policies' clock at its deadline.
    fn fire(&self, time: f64) {
        while self.next_due() <= time {
            let mut timers = self.timers.lock();
            let first = timers.iter_mut().min_by(|a, b| a.due.total_cmp(&b.due));
            let timer = first.expect("a due task");
            determinism::set_now(timer.due);
            (timer.task)();
            timer.due += timer.every;
        }
    }

    /// Ask for the next event until the `+inf` grant, handing each release
    /// the client until it is handled.
    async fn serve(&self, mut client: ClockClient) -> Result<(), ClockError> {
        loop {
            let grant = client.next_event(f64::INFINITY, self.next_due()).await?;
            for message in grant.released {
                self.fire(message.arrival);
                determinism::set_now(message.arrival);
                let take = self.taken(&message.channel, message.seq).await;
                let (back, done) = oneshot::channel();
                let _ = take.send(Turn {
                    arrival: message.arrival,
                    client: Some(client),
                    back: Some(back),
                    cancelled: false,
                });
                client = done
                    .await
                    .unwrap_or_else(|_| fail("a turn ended holding the clock"));
            }
            if grant.time == f64::INFINITY {
                return Ok(());
            }
            self.fire(grant.time);
        }
    }

    /// The frontend LP of the worker at base URL `url`, among those the
    /// outbound `kind` channels reach: the one `WORKERS_ENV` names, or the only one.
    fn frontend(&self, url: Option<&str>, kind: &str) -> String {
        if let Some(lp) = url.and_then(|u| self.workers.get(u)) {
            return lp.clone();
        }
        let reached: Vec<&str> = (self.lookaheads.keys())
            .filter(|ch| self::kind(ch) == kind)
            .filter_map(|ch| ch.strip_prefix("router->")?.split(':').next())
            .collect();
        match reached[..] {
            [only] => only.to_owned(),
            _ => fail(format!(
                "{WORKERS_ENV} names no frontend LP for worker {url:?}"
            )),
        }
    }

    /// The hold point that read (`channel`, `seq`), once it has.
    async fn taken(&self, channel: &str, seq: u64) -> oneshot::Sender<Turn> {
        let key = (channel.to_owned(), seq);
        loop {
            if let Some(take) = self.waiting.lock().remove(&key) {
                return take;
            }
            let wait = tokio::time::timeout(Duration::from_secs(30), self.arrived.notified());
            if wait.await.is_err() {
                warn!("router LP: released {channel} seq {seq} has not reached the router");
            }
        }
    }

    /// Hold a message read on `channel` until it is released.
    async fn hold(&self, channel: &str, seq: u64) -> Turn {
        let (take, turn) = oneshot::channel();
        let key = (channel.to_owned(), seq);
        if self.waiting.lock().insert(key, take).is_some() {
            fail(format!("{channel} seq {seq} arrived twice"));
        }
        self.arrived.notify_one();
        turn.await
            .unwrap_or_else(|_| fail("the clock loop ended with a message held"))
    }

    /// Handle a cancel, posted to `path`, of the request with seq `seq` on
    /// `traffic->router:http`: it goes on to where that request is, if anywhere.
    fn cancel(&self, mut turn: Turn, seq: u64, path: &str) -> Response {
        if let Some(at) = self.requests.lock().get_mut(&seq) {
            at.cancelled = true;
            // On a relay the stamp rides in the body, elsewhere in `tracestate`.
            let stamp = entry(turn.send(&at.channel));
            let body = match kind(&at.channel) {
                "relay" => format!(
                    r#"{{"compass_cancel": {}, "kv_transfer_params": {{"compass": "{stamp}"}}}}"#,
                    at.seq
                ),
                _ => format!(r#"{{"compass_cancel": {}}}"#, at.seq),
            };
            let post = (self.client.post(format!("{}{path}", at.url)))
                .header("content-type", "application/json")
                .header("tracestate", format!("compass={stamp}"))
                .body(body);
            let to = at.channel.clone();
            tokio::spawn(async move {
                if let Err(error) = post.send().await {
                    fail(format!("a cancel on {to} was not delivered: {error}"));
                }
            });
        }
        StatusCode::NO_CONTENT.into_response()
    }
}

/// The seq a cancel's body `{"compass_cancel": <seq>}` names; None for any other body.
fn cancel_seq(body: &[u8]) -> Option<u64> {
    let text = std::str::from_utf8(body).ok()?;
    let seq = text
        .strip_prefix(r#"{"compass_cancel": "#)?
        .strip_suffix('}')?;
    seq.parse().ok()
}

/// The (arrival, seq) of a stamp written `a:<arrival>;s:<seq>` or
/// `a=<arrival> s=<seq>`.
fn stamp(text: &str) -> (f64, u64) {
    let parsed = text
        .strip_prefix("a")
        .and_then(|t| t.get(1..))
        .and_then(|t| t.split_once([';', ' ']))
        .and_then(|(a, s)| Some((a.parse().ok()?, s.get(2..)?.parse().ok()?)));
    parsed.unwrap_or_else(|| fail(format!("malformed compass stamp {text:?}")))
}

/// Ingress layer: a stamped request waits until it is released, and is
/// handled once the router has placed it (`placed`); a cancel,
/// `{"compass_cancel": <its request's seq>}`, at once (`Lp::cancel`). It runs
/// as a task of its own, which the client hanging up does not end.
pub async fn ingress(request: Request, next: Next) -> Response {
    let header = request.headers().get("tracestate");
    let entry = header
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.split(',').find_map(|e| e.trim().strip_prefix("compass=")));
    let Some((lp, seq)) = LP.get().zip(entry.map(|e| stamp(e).1)) else {
        return next.run(request).await;
    };
    let (parts, body) = request.into_parts();
    // A body cut off by a hang-up is read as empty, so it is never placed.
    let body = axum::body::to_bytes(body, usize::MAX)
        .await
        .unwrap_or_default();
    let handled = tokio::spawn(async move {
        let turn = lp.hold("traffic->router:http", seq).await;
        if let Some(cancelled) = cancel_seq(&body) {
            return lp.cancel(turn, cancelled, parts.uri.path());
        }
        let handling = Handling {
            seq,
            turn: Some(turn),
            ..Default::default()
        };
        let request = Request::from_parts(parts, Body::from(body));
        REQUEST
            .scope(RefCell::new(handling), next.run(request))
            .await
    });
    handled
        .await
        .unwrap_or_else(|error| std::panic::resume_unwind(error.into_panic()))
}

/// The request being handled holds the load of the workers at base URLs
/// `prefill` and `decode`: it goes on to that prefill, and is handled.
pub fn placed(prefill: &str, decode: &str) {
    let Some(lp) = LP.get() else {
        return;
    };
    let _ = REQUEST.try_with(|handling| {
        let mut handling = handling.borrow_mut();
        let Some(mut turn) = handling.turn.take() else {
            return;
        };
        let p = lp.frontend(Some(prefill), "http");
        let channel = format!("router->{p}:http");
        let sent = turn.send(&channel);
        handling.stamp = Some(entry(sent));
        handling.pair = Some((p, lp.frontend(Some(decode), "relay")));
        let (url, decode) = (prefill.to_owned(), decode.to_owned());
        let at = Placed {
            channel,
            url,
            seq: sent.1,
            decode,
            ..Default::default()
        };
        lp.requests.lock().insert(handling.seq, at);
    });
}

/// Request header `name`'s `value` as it goes to a worker: a `tracestate`
/// with its `compass` entry restamped for the prefill the request was placed on.
pub fn header(name: &str, value: &str) -> String {
    let stamp = (name == "tracestate")
        .then(|| {
            REQUEST
                .try_with(|h| h.borrow().stamp.clone())
                .ok()
                .flatten()
        })
        .flatten();
    let Some(stamp) = stamp else {
        return value.to_owned();
    };
    let entries = value
        .split(',')
        .map(|e| match e.trim().starts_with("compass=") {
            true => format!("compass={stamp}"),
            false => e.to_owned(),
        });
    entries.collect::<Vec<_>>().join(",")
}

/// Run `task` every `interval_secs` of the LP clock, from now, until `stopped`
/// is set. Without the router LP, hands `task` back.
pub fn periodic<F: Fn() + Send + 'static>(
    interval_secs: u64,
    stopped: &Arc<AtomicBool>,
    task: F,
) -> Result<(), F> {
    let Some(lp) = LP.get() else {
        return Err(task);
    };
    let every = interval_secs as f64;
    lp.timers.lock().push(Timer {
        due: determinism::seconds() + every,
        every,
        stopped: stopped.clone(),
        task: Box::new(task),
    });
    Ok(())
}

/// Hold a prefill response until it is released and restamp its relay entry
/// for the decode the request was placed on; it is handled when the returned
/// turn drops. After its request's cancel it sends nothing on, and the turn
/// says so (`Turn::cancelled`).
pub async fn relay(body: &mut Value) -> Option<Turn> {
    let lp = LP.get()?;
    let entry = body.get_mut("kv_transfer_params")?.get_mut("compass")?;
    let seq = stamp(entry.as_str().unwrap_or_default()).1;
    let (request, pair) =
        (REQUEST.try_with(|h| (Some(h.borrow().seq), h.borrow().pair.clone()))).unwrap_or_default();
    let (p, d) = pair.unwrap_or_else(|| (lp.frontend(None, "http"), lp.frontend(None, "relay")));
    let mut turn = lp.hold(&format!("{p}->router:relay"), seq).await;
    let mut requests = lp.requests.lock();
    let at = request.and_then(|r| requests.get_mut(&r));
    turn.cancelled = at.as_ref().is_some_and(|at| at.cancelled);
    if turn.cancelled {
        return Some(turn);
    }
    let channel = format!("router->{d}:relay");
    let sent = turn.send(&channel);
    *entry = Value::String(self::entry(sent));
    if let Some(at) = at {
        (at.channel, at.url, at.seq) = (channel, std::mem::take(&mut at.decode), sent.1);
    }
    Some(turn)
}

/// The decode stream, each event held until it is released and restamped for
/// traffic. The terminal event drops `guard`, the decode load, before it
/// counts as handled. A task of its own reads it to its end, so a client
/// that hangs up loses only the forwarding. Without the clock: the stream
/// as is, and the guard.
pub fn stream<S>(
    upstream: S,
    guard: WorkerLoadGuard,
) -> (
    BoxStream<'static, Result<Bytes, reqwest::Error>>,
    Option<WorkerLoadGuard>,
)
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let Some(lp) = LP.get() else {
        return (upstream.boxed(), Some(guard));
    };
    let (request, pair) =
        (REQUEST.try_with(|h| (Some(h.borrow().seq), h.borrow().pair.clone()))).unwrap_or_default();
    let decode = pair.map_or_else(|| lp.frontend(None, "relay"), |(_, d)| d);
    let from = format!("{decode}->router:stream");
    let events = futures_util::stream::unfold(
        (upstream.boxed(), Vec::new(), Some(guard)),
        move |(mut upstream, mut buf, mut guard)| {
            let from = from.clone();
            async move {
                loop {
                    if let Some(end) = memmem::find(&buf, b"\n\n") {
                        let event: Vec<u8> = buf.drain(..end + 2).collect();
                        let event = lp.restamp(event, &mut guard, &from, request).await;
                        return Some((Ok(Bytes::from(event)), (upstream, buf, guard)));
                    }
                    match upstream.next().await {
                        Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                        Some(Err(error)) => return Some((Err(error), (upstream, buf, guard))),
                        None if buf.is_empty() => return None,
                        None => {
                            let rest = Bytes::from(std::mem::take(&mut buf));
                            return Some((Ok(rest), (upstream, buf, guard)));
                        }
                    }
                }
            }
        },
    );
    let (forward, forwarded) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut events = events.boxed();
        while let Some(event) = events.next().await {
            let _ = forward.send(event);
        }
    });
    let forwarded = tokio_stream::wrappers::UnboundedReceiverStream::new(forwarded);
    (forwarded.boxed(), None)
}

impl Lp {
    /// Hold one event of the stream from `from` until it is released, and
    /// send it on to traffic; the terminal event ends `request`.
    async fn restamp(
        &self,
        event: Vec<u8>,
        guard: &mut Option<WorkerLoadGuard>,
        from: &str,
        request: Option<u64>,
    ) -> Vec<u8> {
        let line = event.split(|b| *b == b'\n').next().unwrap_or_default();
        let Some(text) = line.strip_prefix(b": compass ") else {
            return event;
        };
        let seq = stamp(&String::from_utf8_lossy(text)).1;
        let mut turn = self.hold(from, seq).await;
        let (arrival, seq) = turn.send("router->traffic:stream");
        let mut out = format!(": compass a={arrival:?} s={seq}").into_bytes();
        out.extend_from_slice(&event[line.len()..]);
        if memmem::find(&out, b"data: [DONE]").is_some() {
            guard.take();
            if let Some(request) = request {
                self.requests.lock().remove(&request);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::stamp;

    #[test]
    fn reads_both_stamp_spellings() {
        assert_eq!(stamp("a:0.000244140625;s:17"), (0.000244140625, 17));
        assert_eq!(stamp("a=1e-07 s=3"), (1e-7, 3));
        assert_eq!(stamp("a=inf s=0"), (f64::INFINITY, 0));
    }
}
