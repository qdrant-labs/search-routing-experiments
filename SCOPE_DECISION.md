# Where the strategy router goes next

Written 2026-08-14/17, after the router verdict. This replaces `SCOPE_BRIEF.md` — delete that file.

This is not a build plan. It records what we now know, what we stopped, the directions worth
considering, and which cheap measurements decide between them. The next step is a design
interview that turns the chosen direction into decisions.

A note on language: everything here is written to be read by someone who has not been in these
sessions. No internal decision numbers, no shorthand. Where a number appears, the population it
was measured on appears with it.

An adversarial review was run against this document on 2026-08-17 and its corrections are folded
in. Three mattered: a failure was attributed to the production classifier when it belonged to our
own router; a precommitted bar was being retired with reasoning that predated the result it
excused; and two of the cheapest directions had no measurement funding them. All three are fixed
below.

---

## What we know

**The shipped router does not work.** It scores slightly *worse* than simply picking one search
mode per collection and never changing it — about 1.5 points worse on a 100-point scale. Two
independent adversarial reviews tried to break that result and could not. Every way of re-cutting
the data gave the same sign.

**The prize is small, and its size depends on which queries you count.** On the current data,
counting every query:

| | Score out of 100 |
|---|---|
| One mode for everything | 56.5 |
| Best single mode per collection | 60.2 |
| Choosing perfectly for every query | 66.0 |

So knowing which collection a query belongs to is worth **3.7 points**, and perfect per-query
choice adds a further **5.8** on top of that.

If you exclude queries that no mode can answer at all — the population our ship bar was set on —
the same three are 68.6, 73.2, and 80.2: knowing the collection worth **4.6**, perfect per-query
choice worth a further **7.0**. Both sets appear because both are quoted in earlier documents; the
difference is the population, not a disagreement.

Published systems that attempt per-query choice capture roughly 4% of such a gap — a fraction of a
point. We agreed in advance to ship only a 2-point gain or better.

**The collection-level number does not mean what it looks like**, for one reason that is measured
and one that is a property of how we built the data:

- Realizing it requires already knowing which mode wins on that collection, which requires
  relevance judgments for it. On a customer's collection nobody has those, so no deployable system
  gets those points either.
- Each collection in our set was deliberately mined for what it is *good at* — legal identifiers
  from one, math problems from another. So each collection's queries are far more uniform than any
  real customer's collection would be. A real documentation site serves both *"how do I configure
  retries"* and *"ERR_CONN_1042"*. This was known and written down before the router was measured;
  it is not a new discovery.

**Four in ten queries score identically under all three modes** — so the choice appears not to
matter for them. That is mostly a measurement limit, not a real tie. When a query has one judged
relevant document the tie rate is 38%; when it has eleven or more, it is under 1%. Nine of the ten
documents each mode returns are different from the others' and were never judged by anyone, so the
score cannot see them.

**About 89% of returned documents carry no human judgment at all** and are counted as irrelevant
whether they are or not. *Caveat added by review:* an independent attempt to rebuild this figure
got 51–53%, because several of the stored result caches are incomplete. The 89% is the recorded
measured value; it is worth re-deriving once before anything leans harder on it.

**We have been comparing a weak version of one mode to a mature version of the other.** Semantic
search here runs a small 2023 embedding model. Keyword search runs BM25, which is not going to
improve. So every result of the form "keyword search won here" is partly a statement about our
embedding model, and an unknown share of them would flip under a modern one. This biases toward
keyword search, and — importantly — it *manufactures* disagreement between the modes, which is
exactly what a router sells. A stronger embedding model would shrink the thing we are trying to
sell.

**The one thing that survives any embedding upgrade** is exact matching on text that embeddings
compress away: version numbers, error codes, product IDs, rare names. `CVE-2021-44228` will never
become a semantic-search win. So a better embedding model does not empty the keyword-wins
category, it *purifies* it — smaller, but cleaner and more learnable.

**One query, to make all of this concrete.** From our data, verified row by row: *"a girl who
can't stop sneezing?"* All three modes put the correct news article first — a report about a Texas
girl sneezing thousands of times a day. Today's answer is therefore "tie — serve the cheapest,
which is keyword search," and that is what the stored decision says. But the query has no codes,
no version numbers, no rare words. It is a plain question. Keyword search only worked because the
article repeats the query's own words. Reword it as *"why does a Texas teenager sneeze thousands of
times a day"* and that overlap mostly disappears. Our current rule serves the fragile mode, on
four in ten queries.

---

## What we stopped

Code stays; nothing runs. Each of these comes back if the direction that needs it is chosen.

- **Setting one search mode per collection, as a product.** A collection is a mixture, so one
  setting is knowingly wrong for part of it.
- **"The best single mode for this collection" as the primary bar.** Not because it was invalid —
  it was chosen deliberately as the *harder* bar, in full knowledge of the uniformity property
  above, and the negative result against it stands. Two forward reasons to change it: no deployed
  system can realize that bar on a collection it has no judgments for, and the thing we have to
  beat commercially is the classifier shipping today. **This is a change of bar going forward, not
  a re-reading of the result.**
- **The planned dataset rebuild in its current form** — fixing the query-feature detectors, then
  re-extracting everything, then re-checking every category. It exists to serve a per-query trained
  model, which is now downstream of several cheaper questions.
- **The query-generation campaign as a way to add rows.** It has produced 714 usable rows out of
  46,856 — about 1.5%. Note what that does and does not say: only the simpler rewriting mode ever
  ran, and the more ambitious mode was never labelled at all, blocked on a review tool that was
  never built. So 1.5% is the yield of what ran, not a verdict on the approach.
- **New query-feature detectors.** Nothing reads them right now.
- **Adding more collections for feature variety.** What the data lacks is queries where the modes
  genuinely disagree, not more variety of query shapes.
- **Re-setting the ship bar for the next model.** Don't re-sign a target for a model whose
  measurement is under repair.

---

## The directions worth considering

Eleven, all global — each changes a premise, rather than fixing something inside the current
approach. They are not mutually exclusive; the grouping section at the end says which go together.

### 1. Give each question its own scoring function

Today one formula does all the work: 70% "was the right document first" plus 30% "how good was the
top ten".

But we are asking three different questions about a query, and only one is answerable by that
formula:

- **What does this query look like it needs?** Answerable from the words alone — a language model
  can do it with no searching and no judgments. Nothing to do with ranking quality.
- **Which mode actually won on this collection?** This is what the current formula measures.
- **Was that win real, or luck of wording?** Answerable by rewording the query and searching
  again — a comparison between two runs, not a score within one.

One formula cannot serve three questions. The interesting information is where the three
**disagree**:

| What it looks like it needs | What actually won | What that row tells us |
|---|---|---|
| semantic | keyword, and it survives rewording | a genuine property of this collection — the hardest and most valuable case |
| semantic | keyword, but it fails after rewording | luck of wording; should never have been recorded as a keyword win |
| keyword | tie, or semantic | the identifier-protection case |
| they agree | they agree | confirms, teaches little |

**What this gates, and what it does not.** It changes what "the right answer" means for every
query, so it gates the directions that depend on per-query labels: direction 4, the learn-to-guess
half of direction 2, and how direction 6 gets evaluated. It does **not** gate directions 7, 8, 9,
or 11 — those are scored against the existing formula or need no labels at all.

Cost note that dictates the order: the first question is cheap — one small model call per query,
no searching. The third is expensive — a rewrite plus three searches. So compute the cheap one
everywhere, find the disagreements, and spend the expensive one only there.

*What would kill it:* if rewording almost never changes the winner, the extra questions add
nothing.

### 2. Let the system see the collection at query time, not just the query

Today the router is allowed the raw query text only. That constraint is why the newer model
architecture is complicated: it has to *learn to guess* properties of the collection it cannot
look at.

The constraint was chosen when we did not know where this would deploy. But in a search engine you
are always querying a specific collection. Statistics computed once when the collection is
indexed, then looked up, are not "inspecting the corpus at query time" — they are a table lookup.

Two refinements:

- **The representation, not the collection.** You never feed documents to a classifier. You feed
  statistics: how rare the query's words are in this collection, how much its vocabulary overlaps,
  how long documents are.
- **Local to the query, not global to the collection.** A collection is a mixture, so an average
  describes no actual query's situation. The useful version asks: *what do the documents nearest
  this query look like?* A documentation site returns a prose neighbourhood for one query and a
  stack-trace neighbourhood for another, from one index.

There is a specific reason this matters. The **newer, never-evaluated model** — the one built after
the shipped router, with side branches — is trained to estimate properties of a query's *correct
answer document*, including how much the query's words overlap it, which is the closest thing we
have to a cause of a keyword win. It has to estimate that because the correct document is unknown
at query time. But the *nearest* documents are not unknown. This measures directly what that model
guesses.

*What would kill it:* if the nearest documents' statistics turn out no more predictive than the
collection's overall statistics, the local version collapses into the global one — and if the
global one predicts nothing either, guessing from the query really is the only option.

*One caution from measured data:* plain word overlap has already been tested and does not separate
keyword wins from semantic wins — median around 0.58 where modes differ, 0.67 where they tie. The
version worth testing weights **rare** words: not whether the query's words appear, but whether
its unusual ones do.

### 3. No trained model at all — a rule over word-frequency statistics

Every design so far assumes the answer is a trained model. It may not be.

A word-frequency table says how common each word is in general English. A query containing a word
that table has never seen — `deadbeefcafe`, `ERR_CONN_1042`, `CVE-2021-44228` — is a query where
exact matching matters and embeddings blur. That is a rule with two or three settings, not a model,
and the table ships as a static file with no runtime dependency. The library is already a project
dependency.

This matters for a reason beyond simplicity. A rule with three settings can be **checked** against
our data. A model with thousands of parameters has to be **learned** from it. Every problem we
found — the unjudged documents, the uniform collections, the weak embedding model — damages
learning far more than it damages checking.

*How this differs from direction 6*, since they overlap: this one is about **what replaces the
model** — a frequency table instead of training. Direction 6 is about **how much you route at
all** — one default plus a protected list. Either can be adopted without the other, and they
compose well.

*Motivating claim, and its status:* a recorded finding says roughly 71% of keyword wins are
invisible to all our query-feature detectors, with word rarity the leading suspect. **The review
could not find any computation behind that number anywhere in the repo** — it appears only as an
assertion in a research note. Treat it as a hypothesis, and note that measurement 2 below tests
the number and the direction at the same time.

*What would kill it:* if the rarity statistics don't separate the cases either — which would point
back at the labels rather than the inputs.

### 4. Build the dataset again, aimed at disagreement

Our set was assembled to cover a wide variety of query shapes. That was the wrong target. What a
router needs is queries where the modes genuinely disagree.

**Be precise about which population that means**, because the two candidate readings differ by
five times: queries where any mode scores differently at all are 49.7% of the set; queries with a
*clear* winner — one mode puts the right document first and the runner-up does not — are 11%,
about 1 in 9. The second is the one worth being dense in.

Direction 1 gives "disagreement" a definition it did not have before: a row is informative exactly
when the three questions give different answers. That is something a collection campaign can aim
at, in a way "cover the feature space" never was.

Two constraints to loosen while doing it: judge more documents per query, since one judged document
per query is what makes four in ten queries look tied; and pull from collections that actually
contain identifier-heavy queries, since ours are mostly general-web questions.

*What would kill it:* if the disagreement categories are genuinely rare in real traffic, a dataset
dense in them trains a model for a distribution nobody sends.

### 5. Decide after searching, not before

Today: look at the query, guess which mode wins, run that one.

Instead: run keyword search first — it needs no neural network, so it is nearly free — and look at
what came back. Did the top result score far above the second? Do the returned documents contain
the query's unusual words? Then decide whether the expensive mode is also needed.

This removes the guess entirely, and the guess is the hard part: whether keyword search works
depends on the words in the *documents*, which are not in the query.

It connects directly to direction 2: the results of the cheap search *are* a sample of the
collection around this query, so you get the collection statistics for free as a side effect.

And it ages the right way. Embedding models keep getting larger and more expensive to run, so
"skip the expensive one when the cheap one already answered" is worth **more** every year — the
opposite of per-query quality routing, whose value shrinks as embeddings improve.

*Testable today at zero cost:* the review confirmed that per-mode raw retrieval scores are stored
for every query at depth ten. So whether a keyword search's own score margin predicts its success
can be answered offline, with no new searching. That is measurement 1.

*What would kill it:* if the cheap search's scores don't separate "I found it" from "I didn't,"
there is no rule to write.

### 6. Protect the queries that break, and route nothing else

Use one default mode for everything, plus a short list of query shapes that must never go to it: a
version number, an error code, a product ID, a name almost nobody has written about. Those go to
keyword search.

This is the narrowest possible product and its value is the value that does not decay — the
identifier cases are permanent. The identifier detectors already built here are exactly the tool
for spotting them.

It is also the fix for the shipped router's clearest single failure: it ranked *"explain
quicksort"* as more keyword-like than semantic, which is plainly wrong. **That was our router, not
the classifier in production** — the production one has never been scored on any query at all,
which is what measurement 4 buys.

*What would kill it:* if these queries are genuinely rare in real traffic, the fix protects almost
nobody.

### 7. Improve the blend, so there is nothing to decide

If one better blend beat all three modes on every collection, routing is pointless — a customer
sets one value and needs no router.

We were supposed to test this and never did. The one collection we probed said the current blend
already beats the alternatives we had deleted, and that tuning its parameter is worth almost
nothing. But that was one collection with a hand-written implementation rather than the engine's
own, so it is a hint, not an answer.

*What would kill it:* nothing kills it — its *success* is what kills the product, and not only the
routing version. If one fixed blend wins everywhere, per-region configuration and a
customer-facing tool are equally unnecessary. That is why it runs early.

### 8. Improve the embedding model instead of the router

The whole program treats the search stack as fixed and the routing decision as the variable. That
may be backwards. A modern embedding model may well gain more than a *perfect* router could ever
win — perfect per-query choice is worth 5.8 points on all queries, 7.0 on answerable ones — and
the two compete for the same engineering time.

This is not an argument against routing. It is an argument that the comparison has never been
made, and it should be, because it is the first question a skeptical reader asks.

*What would kill it:* if a modern model gains little on our collections — which would itself be
surprising and worth knowing.

### 9. Ship the measuring tool, not the decision

Instead of an automatic router, give the customer an instrument: point it at your collection, it
samples your queries, runs all three modes, shows you where each wins, and you set the policy.

This is honest about how uncertain the automatic version is, requires no per-query model, and is
shippable far sooner than anything else here. It also generates exactly the data an automatic
version would later need.

*What would kill it:* if customers won't do the work, or won't supply queries to sample.

### 10. Widen what the system can output

Today the choice is three options: keyword, semantic, or a fixed even blend. Instead, choose the
blend ratio per query.

**Corrected by review — the earlier framing here was backwards.** On the one collection tested,
adding finer ratios raised the best achievable score by 2.6 points, while getting the three-way
choice right was worth 5.0. So finer ratios widen the ceiling by roughly *half* what the existing
three-way problem is worth. This is an addition to that problem, not a replacement for it.

*What would kill it:* the extra ceiling was measured once, on one collection. And it is the most
expensive to evaluate: we stored only the top ten documents per search, and changing the blend
reshuffles what sits below rank ten, so every search must be re-run and recorded deeper.

### 11. Publish the dataset

46,142 queries across 42 collections, each measured against all three search modes, plus the
evaluation setup that produced the verdict. Nobody has published a set like it. It has value
whether or not any router ships.

*What would kill it:* nothing, but it requires stating the known flaws honestly — the unjudged
documents, the embedding-model pinning — and some are unflattering.

### A twelfth, speculative

**Change the query instead of choosing the retriever** — expand it for semantic search, or pull
the identifier forward for keyword search. A different action entirely. Listed because it is
global, not because it is ready.

---

## What to measure first, and why in that order

Eight measurements. The first two are free and were missing from the earlier draft — the review
caught that the two cheapest directions had nothing funding them.

1. **Does the cheap search's own score tell you when it succeeded?** Free, offline, no new
   searching: per-mode raw scores are stored for every query at depth ten. Test whether the
   keyword score margin — top result versus runner-up — separates the queries where keyword search
   found the answer from those where it didn't. This is direction 5's entire mechanism, answerable
   this week.
2. **Does word rarity separate keyword wins?** Free, offline: the frequency library is already a
   dependency and the query text is on disk. Test whether rare or unknown words in a query predict
   a keyword win. This tests direction 3 *and* checks the unverified "71% invisible" claim in one
   pass.
3. **Judge more documents on two collections.** Pick two where queries have one judged document
   each and judge deeper. This is what tells us whether four-in-ten ties are real or a measurement
   limit — and it re-derives the 89% figure the review could not reproduce.
4. **Score the live classifier — on the held-out queries, then on all of them.** Its result cache
   is empty; we have never measured the thing we are trying to replace. The held-out set is one
   afternoon. Scoring *everything* costs about five times more and hands us direction 1's cheap
   question on every row as a by-product of the same purchase.
5. **Swap in a modern embedding model on three collections.** Re-run only the semantic side;
   keyword results and judgments are unchanged. Report one number: how many keyword wins flip to
   semantic, split by whether the query carries an identifier or rare term. This prices the decay
   and tests whether the supposedly-invisible keyword wins are really embedding failures.
6. **The rewording test**, on queries where the first two questions disagree. A rewrite plus three
   searches per query, so scope it to disagreements only.
7. **Collection statistics.** There is a script for this, written and never run: predict each
   collection's best mode from six statistics about it, holding one collection out at a time. Read
   it strictly as a *negative* detector — if it can't beat guessing the most common answer, that
   form is closed. If it isn't closed, spike the local version: cluster three collections, profile
   each cluster, and see whether the nearest cluster predicts better than the collection average.
8. **Time the three modes.** One evening. "Keyword search is cheaper" has been an assumption since
   it was written down.

**The rule that orders these** is about what survives a change to our search stack:

- **A judgment on a (query, document) pair is permanent.** It is a fact about the collection and
  the query. It survives an embedding swap, a blend change, a re-index. Buy it once, it keeps
  paying — which is why deeper judgments come early despite costing money.
- **Anything computed from what a *semantic* search returned is pinned to today's model.** Which
  mode's results look better, and the statistics of the nearest documents, both depend on the
  embedding model that produced them. Buy either before the swap and you have precisely measured a
  stack you are about to retire.
- **Refinement from review:** keyword-side statistics — word rarity, keyword score margins — depend
  on the tokenizer and the collection's term counts, not on the embedding model. **They do not
  wait for the swap.** That is why measurements 1 and 2 can run immediately.

So measurement 6 and the local half of 7 wait for 5, and run on the same collections with the same
models so they describe one system.

---

## Which directions go together

**Look instead of guessing.** Directions 1, 5, 6, and the local half of 2. Fix what the right
answer means, then ship rules that read evidence: default to semantic, protect identifier queries,
skip the expensive search when the cheap one already answered. Cheapest, ships soonest, needs the
query-only constraint relaxed, and leaves the trained-model work unused.

**Finish the model.** Directions 1, 4, 10, and the learn-to-guess half of 2. Fix what the right
answer means, rebuild the dataset aimed at disagreement, widen the output to a blend ratio, and
keep teaching the model to estimate what it cannot see. Keeps the built architecture as the
product; needs the two most expensive measurements before it can be scored at all.

**Don't route — configure.** Directions 2 and 9. Decide per region of a collection, or hand the
decision to the customer with an instrument. No per-query model anywhere.

**Outside all three groups:**

- **Direction 7** is not a member of any group — its success would make all three moot, not just
  the routing one. It is the cheapest way to find out whether this whole problem exists, so it runs
  early.
- **Direction 8** is the question a skeptic asks first: would upgrading the embedding model beat
  all of this?
- **Direction 11** pays regardless of which group is chosen.
- **Direction 3** is the cheapest possible baseline and is worth testing inside any group. If a
  frequency table works, most of the rest is unnecessary.

---

## Open questions the design interview must settle

1. **May the router see collection statistics at query time?** This is the root question — it
   decides the shape of everything else. Keep the query-only rule and directions 2, 5, and 6 are
   all dead by the letter of it, while the built architecture stays primary. Relax it and most of
   this list opens up. The fact that forces the question: in a search engine, the collection is
   always known at query time.
2. **If looking up the collection's statistics works, what happens to the part of the model built
   to guess them?** They do the same job. Keeping both is defensible — the guess as a fallback
   where no statistics exist — but neither being named primary is how two mechanisms quietly
   compete.
3. **Which embedding model for the swap?** It should preview what a customer will run in two years,
   not be the next size up from what we have. A half-step could return an ambiguous number and
   settle nothing.
4. **Does a reworded query keep its original judgments?** Our existing rewrites — typos, casing,
   word order, politeness — all keep the content words, so the original document still answers. A
   rewrite that deliberately *removes* word overlap does not obviously inherit that, and the whole
   rewording test depends on it.
5. **How is a three-part answer stored, and what does the serving rule become?** Today, when modes
   tie, we serve the cheapest — which means keyword search, the fragile one, on four in ten
   queries. Worth noting what the review confirmed: nothing *trains* on that column, so this is a
   serving-policy problem, not a contaminated-labels problem.
6. **What is the bar now?** The proposal is the live classifier, at the same 2-point margin, chosen
   as a forward decision rather than as a re-reading of the old result. It needs the same
   commit-in-advance discipline the first bar got.
7. **Is "what does this query look like it needs" a recorded fact, an input to a model, or a
   baseline to beat?** Today it is only a baseline. If it becomes part of the answer, the rewording
   test is what keeps it honest — otherwise it is one model's opinion, checkable against nothing.
8. **The query-feature detectors have 92 failing round-trip tests.** Repaired, or parked with the
   dataset rebuild they block?

---

## What comes next

A design interview with this document as the starting plan, working through the open questions
above — question 1 first, since the others depend on it — and ending in a written specification.

Two decisions here are hard to reverse and driven by real trade-offs, so they are worth recording
with their reasoning: changing the bar we measure against, and treating the answer as three
separately-scored questions rather than one.
