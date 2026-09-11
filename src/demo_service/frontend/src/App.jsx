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
  { id: "welcome", label: "01 Budget" },
  { id: "collection", label: "02 Collection" },
  { id: "setup", label: "03 Test setup" },
  { id: "candidate", label: "04 Candidate" },
  { id: "retrieval", label: "05 Retrieval" },
  { id: "suite", label: "06 Suite" },
];

const Logo = () => (
  <svg width="22" height="24" viewBox="0 0 22 24" fill="none">
    <path d="M11 0L21.4 6v12L11 24 .6 18V6L11 0z" fill="#DC244C" />
    <path d="M11 5.2l6 3.4v6.8l-6 3.4-6-3.4V8.6l6-3.4z" fill="#0D1422" />
    <path d="M11 9.1l2.6 1.5v3l-2.6 1.5-2.6-1.5v-3L11 9.1z" fill="#DC244C" />
  </svg>
);

function BurnDown({ run, budget }) {
  const cap = run?.budget_usd ?? budget.budget_usd;
  const remaining = run?.remaining_usd ?? budget.remaining_usd ?? cap;
  const pct = Math.max(0, Math.min(100, (remaining / cap) * 100));
  return (
    <span className="burndown" title={`$${remaining.toFixed(4)} of $${cap} left`}>
      <span className="burndown-bar">
        <span className="burndown-fill" style={{ width: `${pct}%` }} />
      </span>
      ${remaining.toFixed(2)} left
    </span>
  );
}

function WelcomeScreen({ health, busy, onStart }) {
  const presets = health?.budget_presets ?? [0.5, 2.0, 5.0];
  const dflt = health?.default_budget ?? 2.0;
  return (
    <section className="welcome">
      <div className="eyebrow">Before we start · set a spend cap</div>
      <h1>How much is this session allowed to spend?</h1>
      <p className="lede">
        Generating a query costs about $0.0003; judging a pooled result set
        about $0.008. This cap is the only limit — nothing in the session can
        spend past it. $2 comfortably covers a full walk-through.
      </p>
      <div className="budget-chips">
        {presets.map((p) => (
          <button
            key={p}
            className={`btn budget-chip${p === dflt ? " primary" : ""}`}
            disabled={busy}
            onClick={() => onStart(p)}
          >
            ${p.toFixed(2)}
            {p === dflt && <span className="chip-note">recommended</span>}
          </button>
        ))}
      </div>
    </section>
  );
}

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
  const [screen, setScreen] = useState("welcome");
  const [budget, setBudget] = useState(null);
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

  const startSession = (budgetUsd) =>
    act(async () => {
      const s = await post("/session", { budget_usd: budgetUsd });
      setBudget(s);
      setScreen("collection");
    });

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
      setBudget(null);
      setScreen("welcome");
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
          {budget && <BurnDown run={run} budget={budget} />}
          <button className="btn" style={{ padding: "5px 12px", fontSize: 12 }} onClick={reset}>
            Start over
          </button>
        </div>
      </header>
      <main>
        {error && <div className="error">{error}</div>}
        {screen === "welcome" && (
          <WelcomeScreen health={health} busy={busy} onStart={startSession} />
        )}
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
