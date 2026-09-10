// The five presentation screens, matched to the approved design canvas.
// Pure render components — every state change goes through App's handlers.

export function Chip({ accent, children }) {
  return <span className={`chip${accent ? " accent" : ""}`}>{children}</span>;
}

const IdentifierIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <rect x="2.75" y="4.75" width="14.5" height="10.5" rx="2" stroke="#DC244C" strokeWidth="1.5" />
    <path d="M5.5 8h5M5.5 11h9" stroke="#DC244C" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

const MessyIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <path d="M3 10h4l2-5 3 10 2-5h3" stroke="#DC244C" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const ChatIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <path
      d="M3 5.5A2.5 2.5 0 015.5 3h9A2.5 2.5 0 0117 5.5v5a2.5 2.5 0 01-2.5 2.5H8l-3.6 3.2c-.5.4-1.4.1-1.4-.6V5.5z"
      stroke="#DC244C"
      strokeWidth="1.5"
    />
  </svg>
);

export function CollectionScreen({ source, selectedDocId, onSelect, onContinue }) {
  if (!source) return <div className="card hint">loading collection…</div>;
  return (
    <section>
      <div className="eyebrow">
        124,428 documents · 0 queries · 0 answer key
      </div>
      <h1>You have documents. No queries, no answer key.</h1>
      <p className="lede">
        Pick a source document — the chips show what the query taxonomy already
        detects in each one. Your first search test will be grounded in the
        document you choose.
      </p>
      <div className="shelf">
        {source.documents.map((d) => (
          <button
            key={d.doc_id}
            className={`doc-card${d.doc_id === selectedDocId ? " selected" : ""}`}
            onClick={() => onSelect(d.doc_id)}
          >
            <span className="head">
              <Chip>doc {d.doc_id}</Chip>
              {d.doc_id === selectedDocId && <span className="selected-tag">SELECTED</span>}
            </span>
            <span className="title">{d.title}</span>
            <span className="excerpt">{d.excerpt}…</span>
            {d.identifier_surface ? (
              <span className="chip-row">
                <Chip accent>
                  {d.identifier_surface} · {d.identifier_feature}
                </Chip>
              </span>
            ) : (
              <span className="none">no identifier detected</span>
            )}
          </button>
        ))}
      </div>
      <div className="actions end" style={{ marginTop: 18 }}>
        <button className="btn primary" onClick={onContinue}>
          Set up the test →
        </button>
      </div>
    </section>
  );
}

export function SetupScreen({ source, doc, busy, onChoose }) {
  if (!doc) return <div className="card hint">pick a document first</div>;
  const behaviors = Object.fromEntries(source.behaviors.map((b) => [b.id, b]));
  const supports = (id) => doc.behaviors.includes(id);
  return (
    <section>
      <div className="eyebrow">Step 2 · one document, one behavior</div>
      <h1>What should this test ask for?</h1>
      <div className="card">
        <div className="chip-row" style={{ marginBottom: 8 }}>
          <span className="eyebrow">Source document</span>
          <Chip>doc {doc.doc_id}</Chip>
          {doc.identifier_surface && (
            <Chip accent>
              {doc.identifier_surface} · {doc.identifier_feature}
            </Chip>
          )}
        </div>
        <h2>{doc.title}</h2>
        <p className="lede" style={{ marginBottom: 0 }}>{doc.excerpt}…</p>
      </div>
      <div className="behaviors three" style={{ marginTop: 16 }}>
        <div className="card behavior-card">
          <div className="head">
            <IdentifierIcon />
            <span>{behaviors.identifier?.label}</span>
          </div>
          <div className="desc">{behaviors.identifier?.description}</div>
          {doc.identifier_surface ? (
            <div className="example">
              e.g. planer wheel <span className="hot">{doc.identifier_surface}</span>
            </div>
          ) : (
            <div className="example">
              e.g. security bar <span className="hot">no</span> drilling
            </div>
          )}
          <div className="hint" style={{ marginBottom: 4 }}>
            The system rolls the structure — {(doc.structured_options || [])
              .map((o) => o.label.split(" — ")[0])
              .join(", ")} — and the gate verifies whatever it rolled.
          </div>
          <div className="actions">
            <button
              className="btn primary"
              disabled={busy}
              onClick={() => onChoose("identifier", null)}
            >
              Generate →
            </button>
          </div>
        </div>
        <div className="card behavior-card">
          <div className="head">
            <ChatIcon />
            <span>{behaviors.conversational?.label}</span>
          </div>
          <div className="desc">{behaviors.conversational?.description}</div>
          <div className="example">hey, i&rsquo;m looking for … if you have one</div>
          <div className="actions">
            <button
              className="btn primary"
              disabled={busy}
              onClick={() => onChoose("conversational", null)}
            >
              Generate →
            </button>
          </div>
        </div>
        <div className="card behavior-card">
          <div className="head">
            <MessyIcon />
            <span>{behaviors.real_traffic?.label}</span>
          </div>
          <div className="desc">{behaviors.real_traffic?.description}</div>
          <div className="example">makita <span className="hot">whel</span> para concreto</div>
          <div className="actions">
            <button
              className="btn primary"
              disabled={busy}
              onClick={() => onChoose("real_traffic", null)}
            >
              Generate →
            </button>
          </div>
        </div>
      </div>
      {busy && <div className="hint" style={{ marginTop: 12 }}>generating…</div>}
    </section>
  );
}

function ShapeChecks({ checks }) {
  return (
    <div className="card check-list">
      {(checks.shape_checks || []).map((chk, i) => (
        <div key={i} className="check-row">
          <span className="label">
            <span className={chk.passed ? "pass" : "fail"}>{chk.passed ? "✓" : "✗"}</span>
            <span>{chk.target}</span>
          </span>
          <span className="measured">measured {chk.measured}</span>
        </div>
      ))}
      <div className="check-row">
        <span className="label">
          <span className={checks.guard_rejected ? "fail" : "pass"}>
            {checks.guard_rejected ? "✗" : "✓"}
          </span>
          <span>
            {checks.guard_rejected
              ? `rejected: ${checks.guard_reason}`
              : "generation guards — not a refusal, duplicate, or verbatim lift"}
          </span>
        </span>
        <span className="measured">{checks.guard_rejected ? "" : "clean"}</span>
      </div>
      <div className="check-row">
        <span className="label">
          <span className={checks.answerable_status === "answers" ? "pass" : ""}>
            {checks.answerable_status === "answers" ? "✓" : "·"}
          </span>
          <span>
            source answerability:{" "}
            {checks.answerable_status ? (
              <strong>{checks.answerable_status}</strong>
            ) : (
              "not yet inspected"
            )}
          </span>
        </span>
        <span className="measured">{checks.answerable_status ? checks.answerable_method : ""}</span>
      </div>
    </div>
  );
}

export function CandidateScreen({ run, doc, busy, onRepair, onAnswerable, onRetrieve }) {
  if (!run) return <div className="card hint">no candidate yet — choose a behavior</div>;
  const checks = run.checks;
  const shapeFailed = checks && !checks.shape_passed;
  return (
    <section>
      <div className="two-col">
        <div>
          <div className="eyebrow">
            Generated query · {run.behavior_id}
            {run.features?.length ? ` · rolled: ${run.features.join(" + ")}` : ""}
            {" · "}
            {run.mode === "replay" ? "prepared replay" : "live"}
          </div>
          <div className="card query-hero">&ldquo;{run.query_text}&rdquo;</div>
          {run.fallback_reason && (
            <div className="fallback-note">fallback: {run.fallback_reason}</div>
          )}
          {checks ? (
            <ShapeChecks checks={checks} />
          ) : (
            <div className="card hint">not checked yet</div>
          )}
          {shapeFailed && (
            <div className="card">
              <div className="hint" style={{ marginBottom: 8 }}>
                A shape check failed — one repair attempt is allowed, same budget.
              </div>
              <button className="btn" disabled={busy} onClick={onRepair}>
                Repair (one attempt)
              </button>
            </div>
          )}
          <div className="actions end" style={{ marginTop: 14 }}>
            <button className="btn primary" disabled={busy} onClick={onRetrieve}>
              Run retrieval →
            </button>
          </div>
        </div>
        <div>
          {doc && (
            <div className="card">
              <div className="chip-row" style={{ marginBottom: 8 }}>
                <span className="eyebrow">Source document</span>
                <Chip>doc {doc.doc_id}</Chip>
              </div>
              <h2 style={{ fontSize: 16 }}>{doc.title}</h2>
              <p className="lede" style={{ marginBottom: 0, fontSize: 13 }}>{doc.excerpt}…</p>
            </div>
          )}
          <div className="card inspection">
            <h3>Your inspection</h3>
            <div className="note">
              Does the source document answer this query? Your verdict is
              recorded on the test — nothing regenerates. A test is only saved
              once you confirm it.
            </div>
            <div className="actions">
              <button className="btn good" disabled={busy} onClick={() => onAnswerable("answers")}>
                ✓ Source answers it
              </button>
              <button className="btn" disabled={busy} onClick={() => onAnswerable("does_not_answer")}>
                ✗ It does not
              </button>
            </div>
          </div>
        </div>
      </div>
      {busy && <div className="hint">working…</div>}
    </section>
  );
}

export function RetrievalScreen({ run, fetchLimit }) {
  if (!run?.retrieval) return <div className="card hint">retrieval has not run</div>;
  return (
    <section>
      <div className="eyebrow">Same query · three strategies · fetch depth {fetchLimit}</div>
      <h1 style={{ fontSize: 30 }}>Where does each strategy put the source?</h1>
      <div className="query-hero" style={{ padding: "4px 0 18px", fontSize: 15, color: "var(--muted)" }}>
        &ldquo;{run.query_text}&rdquo;
      </div>
      <div className="results">
        {run.retrieval.map((strat) => (
          <div key={strat.strategy} className="card result-col">
            <div className="head">
              <span className="name">{strat.strategy.replace("_only", "").replace("pure_rrf", "hybrid rrf")}</span>
              <span className={`rank-badge ${strat.source_rank ? "found" : "missing"}`}>
                {strat.timed_out
                  ? "timed out"
                  : strat.source_rank
                    ? `source #${strat.source_rank}`
                    : `not in top ${fetchLimit}`}
              </span>
            </div>
            {strat.ranked_doc_ids.slice(0, 8).map((id, i) => {
              const isSource = strat.source_rank === i + 1 && id === run.doc_id;
              const title = strat.ranked_titles?.[i] || id;
              return (
                <div key={id} className={`rank-row${isSource ? " source" : ""}`}>
                  <span className="n">#{i + 1}</span>
                  <span className="t">{title}</span>
                  {isSource && <span className="tag">SOURCE</span>}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="footnote">
        The source is a known answer after inspection, not the only relevant
        document. Absent means &ldquo;not found in top {fetchLimit}&rdquo;, never
        &ldquo;no relevant result&rdquo;.
      </div>
    </section>
  );
}

export function RankBadges({ retrieval }) {
  if (!retrieval) return null;
  return (
    <span className="chip-row">
      {retrieval.map((st) => (
        <span
          key={st.strategy}
          className={`rank-badge ${st.source_rank ? "found" : "missing"}`}
        >
          {st.strategy.replace("_only", "").replace("pure_rrf", "rrf")}{" "}
          {st.source_rank ? `#${st.source_rank}` : "—"}
        </span>
      ))}
    </span>
  );
}

export function SuiteScreen({ saved, suite, saveError }) {
  return (
    <section>
      <div className="eyebrow">Retained · rerunnable against any retrieval config</div>
      <h1 style={{ fontSize: 32 }}>Your first inspectable search tests</h1>
      {saveError && <div className="error">Not saved: {saveError}</div>}
      {saved && (
        <div className={`card saved-card${saved.accepted ? "" : " rejected"}`}>
          <div className={`savedline${saved.accepted ? "" : " warn"}`}>
            {saved.accepted
              ? "✓ Saved as a reusable test"
              : "◦ Retained as a rejected attempt — feedback, not a test"}
          </div>
          {saved.rejected_reason && (
            <div className="hint" style={{ marginBottom: 8 }}>{saved.rejected_reason}</div>
          )}
          <div className="q">&ldquo;{saved.query_text}&rdquo;</div>
          <div className="chip-row">
            <Chip>{saved.behavior_id}</Chip>
            {(saved.provenance?.features || []).map((f) => <Chip key={f}>{f}</Chip>)}
            <Chip>doc {saved.doc_id}</Chip>
            <Chip>{saved.mode}</Chip>
            <Chip>${saved.provenance?.spent_usd ?? 0} spend</Chip>
            <RankBadges retrieval={saved.retrieval} />
          </div>
        </div>
      )}
      <div className="eyebrow" style={{ margin: "20px 0 6px" }}>Suite so far</div>
      {suite.length === 0 ? (
        <div className="hint">nothing saved yet</div>
      ) : (
        suite.map((r) => (
          <div key={r.query_id} className="card suite-row">
            <div className="row-top">
              <span className="q">&ldquo;{r.query_text}&rdquo;</span>
              <span className={`chip ${r.accepted ? "ok" : "warn"}`}>
                {r.accepted ? "accepted" : "rejected attempt"}
              </span>
            </div>
            <div className="row-bottom">
              <span className="chip-row">
                <Chip>{r.behavior_id}</Chip>
                {(r.provenance?.features || []).map((f) => <Chip key={f}>{f}</Chip>)}
                <Chip>doc {r.doc_id}</Chip>
              </span>
              <RankBadges retrieval={r.retrieval} />
            </div>
          </div>
        ))
      )}
      <div className="closing">
        Bring your collection.{" "}
        <span className="hot">Leave with your first inspectable search tests.</span>
      </div>
    </section>
  );
}
