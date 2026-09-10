import { useCallback, useEffect, useState } from "react";
import { get, post } from "./api.js";
import {
  CandidateScreen,
  CollectionScreen,
  RetrievalScreen,
  SetupScreen,
  SuiteScreen,
} from "./screens.jsx";

const STEPS = [
  { id: "collection", label: "01 Collection" },
  { id: "setup", label: "02 Test setup" },
  { id: "candidate", label: "03 Candidate" },
  { id: "retrieval", label: "04 Retrieval" },
  { id: "suite", label: "05 Suite" },
];

const Logo = () => (
  <svg width="22" height="24" viewBox="0 0 22 24" fill="none">
    <path d="M11 0L21.4 6v12L11 24 .6 18V6L11 0z" fill="#DC244C" />
    <path d="M11 5.2l6 3.4v6.8l-6 3.4-6-3.4V8.6l6-3.4z" fill="#0D1422" />
    <path d="M11 9.1l2.6 1.5v3l-2.6 1.5-2.6-1.5v-3L11 9.1z" fill="#DC244C" />
  </svg>
);

function ModeChip({ health }) {
  if (!health) return <span className="mode-chip down"><span className="dot" />SERVICE DOWN</span>;
  if (health.mode === "live") return <span className="mode-chip live"><span className="dot" />LIVE</span>;
  return (
    <span className="mode-chip replay" title={health.reason || ""}>
      <span className="dot" />PREPARED REPLAY
    </span>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [source, setSource] = useState(null);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [run, setRun] = useState(null);
  const [saved, setSaved] = useState(null);
  const [saveError, setSaveError] = useState(null);
  const [suite, setSuite] = useState([]);
  const [screen, setScreen] = useState("collection");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await get("/health"));
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    get("/source")
      .then((s) => {
        setSource(s);
        setSelectedDocId(s.default_doc_id);
      })
      .catch((e) => setError(e.message));
  }, [refreshHealth]);

  const doc = source?.documents.find((d) => d.doc_id === selectedDocId) || null;

  // One wrapper for every stage action: busy flag, error surface, banner
  // refresh (a stage can flip the run to replay mid-demo).
  const act = useCallback(
    async (fn, nextScreen) => {
      setBusy(true);
      setError(null);
      try {
        const result = await fn();
        if (result) setRun(result);
        if (nextScreen) setScreen(nextScreen);
      } catch (e) {
        setError(e.message);
      } finally {
        setBusy(false);
        refreshHealth();
      }
    },
    [refreshHealth]
  );

  const choose = (behaviorId, feature) =>
    act(async () => {
      await post("/run/generate", {
        behavior: behaviorId,
        doc_id: selectedDocId,
        ...(feature ? { feature } : {}),
      });
      return post("/run/check");
    }, "candidate");

  const repair = () => act(() => post("/run/repair"));
  const answerable = (status) => act(() => post("/run/answerable", { status }));
  const retrieve = () => act(() => post("/run/retrieve"), "retrieval");

  const save = () =>
    act(async () => {
      setSaveError(null);
      setSaved(null);
      try {
        setSaved(await post("/run/save"));   // accepted or retained-rejected
      } catch (e) {
        setSaveError(e.message);             // only "nothing checked yet"
      }
      setSuite(await get("/suite"));
      setScreen("suite");
    });

  const reset = () =>
    act(async () => {
      await post("/run/reset");
      setRun(null);
      setSaved(null);
      setSaveError(null);
      setScreen("collection");
    });

  const stepIndex = STEPS.findIndex((s) => s.id === screen);
  const goto = (id) => {
    if (id === "suite") return save();
    setScreen(id);
  };

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <Logo />
          <span className="brand-name">search-test demo</span>
          <span className="brand-sub">{source?.collection ?? ""} collection</span>
        </div>
        <nav className="stepper">
          {STEPS.map((s, i) => (
            <button
              key={s.id}
              className={`step${i === stepIndex ? " active" : i < stepIndex ? " done" : ""}`}
              onClick={() => goto(s.id)}
            >
              {s.label}
            </button>
          ))}
        </nav>
        <div className="status">
          <ModeChip health={health} />
          <span className="spend">
            ${run?.spent_usd ?? "0.0000"} · {run?.calls_used ?? 0}/2 calls
          </span>
          <button className="btn" style={{ padding: "5px 12px", fontSize: 12 }} onClick={reset}>
            Start over
          </button>
        </div>
      </header>
      <main>
        {error && <div className="error">{error}</div>}
        {screen === "collection" && (
          <CollectionScreen
            source={source}
            selectedDocId={selectedDocId}
            onSelect={setSelectedDocId}
            onContinue={() => setScreen("setup")}
          />
        )}
        {screen === "setup" && (
          <SetupScreen source={source} doc={doc} busy={busy} onChoose={choose} />
        )}
        {screen === "candidate" && (
          <CandidateScreen
            run={run}
            doc={doc}
            busy={busy}
            onRepair={repair}
            onAnswerable={answerable}
            onRetrieve={retrieve}
          />
        )}
        {screen === "retrieval" && (
          <RetrievalScreen run={run} fetchLimit={source?.fetch_limit ?? 20} />
        )}
        {screen === "suite" && (
          <SuiteScreen saved={saved} suite={suite} saveError={saveError} />
        )}
      </main>
    </>
  );
}
