"""
Benchmark of backend/services/verification.py (VerificationEngine) on the 53 chunks that
were actually indexed in storage/qdrant (CropChain repo).

Positive class = "hallucinated claim set" (the thing the verifier is supposed to catch).
A case is FLAGGED if decision != VERIFIED (WARNING or REJECTED).

Usage: python3 -I verif_bench.py <project_dir> <chunks_dump.json> [out.json]
Run it under different PYTHONHASHSEED values to expose non-determinism.
"""
import sys, json, random, collections

project, dump = sys.argv[1], sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else None
sys.path.insert(0, project)
from backend.services.verification import VerificationEngine

chunks = [x["payload"] for x in json.load(open(dump))["repo_3e9c5a1753fa"]]
by_fqn = {c["fqn"]: c for c in chunks}
engine = VerificationEngine(chunks=chunks, repo_id="bench")
rng = random.Random(7)


def context_for(fqn, k=5):
    """Retrieved context = target chunk + 4 random others (what a top-5 rerank would hand over)."""
    others = [c for c in chunks if c["fqn"] != fqn]
    return [by_fqn[fqn]] + rng.sample(others, k - 1)


def claims(symbols=(), structures=(), edges=(), assertions=()):
    return {"symbols_mentioned": list(symbols), "structures": list(structures),
            "call_edges": list(edges), "factual_assertions": list(assertions)}


cases = []  # (type, label[1=hallucinated], target_fqn, claims)

# ---------- FAITHFUL ----------
for c in chunks:  # F_struct: symbol + correct signature
    cases.append(("faithful:symbol+structure", 0, c["fqn"],
                  claims([c["fqn"]], [{"symbol": c["fqn"], "claimed_params": c["params"],
                                       "claimed_parent": c["parent"]}])))

P, A, S, T, U = ("routes/paymentRoutes.js", "routes/adminRoutes.js", "public/script.js",
                 "public/test-admin.js", "routes/authRoutes.js")
true_edges = [  # (caller fqn, callee as an LLM would write it, kind)
    (f"{P}::POST /create-order", f"{P}::cashfreeRequest", "same-file call"),
    (f"{P}::POST /verify-payment", f"{P}::cashfreeRequest", "same-file call"),
    (f"{S}::loadCrops", f"{S}::createCropCard", "same-file call"),
    (f"{S}::createCropCard", f"{S}::esc", "same-file call"),
    (f"{T}::farmerRow", f"{T}::statusBadge", "same-file call"),
    (f"{T}::farmerRow", f"{T}::fmt", "same-file call"),
    (f"{T}::statusBadge", f"{T}::esc", "same-file call (name defined in 2 files)"),
    (f"{T}::farmerRow", f"{T}::esc", "same-file call (name defined in 2 files)"),
    (f"{A}::GET /stats", f"{A}::requireAdmin", "middleware reference"),
    (f"{A}::GET /users", f"{A}::requireAdmin", "middleware reference"),
    (f"{U}::POST /login", "User.findOne", "external/model call"),
    (f"{A}::requireAdmin", "User.findById", "external/model call"),
    (f"{P}::cashfreeRequest", "fetch", "external/builtin call"),
    ("routes/analyticsRoutes.js::GET /:crop", "analyticsController.getStats", "cross-file (callee not indexed)"),
    ("server.js::POST /api/list-crop", "cropRoutes", "cross-file (callee not indexed)"),
]
for caller, callee, kind in true_edges:
    cases.append((f"faithful:true call edge [{kind}]", 0, caller,
                  claims([caller], edges=[{"caller": caller, "callee": callee}])))

true_assert = [
    (f"{U}::POST /login", "Asserts that the login route finds a user by email and password and returns 401 when no user matches."),
    (f"{U}::POST /signup", "Asserts that signup returns 409 when an account with the same email already exists."),
    (f"{U}::POST /signup", "Asserts that new users are created with status Pending unless the role is Admin."),
    (f"{A}::requireAdmin", "Asserts that requireAdmin reads the x-user-id header and returns 403 if the user is not an Admin."),
    (f"{A}::GET /stats", "Asserts that the stats route counts users, farmers, buyers, crops, tools and prices using countDocuments."),
    (f"{P}::POST /create-order", "Asserts that create-order builds an order payload in INR and calls cashfreeRequest to create the order."),
    (f"{P}::POST /verify-payment", "Asserts that verify-payment fetches the order status from Cashfree and saves or updates a Transaction."),
    (f"{P}::cashfreeRequest", "Asserts that cashfreeRequest sends a fetch request with the x-client-id and x-client-secret headers and throws an error when the response is not ok."),
    (f"{P}::GET /farmer/:identifier", "Asserts that the route returns completed payments for a farmer along with total earnings and total sales."),
    ("routes/cropRoutes.js::GET /", "Asserts that the route returns all crops sorted by newest first."),
    ("routes/toolRoutes.js::GET /", "Asserts that the tools list can be filtered by category and searched by toolName, village or ownerName."),
    (f"{S}::loadCrops", "Asserts that loadCrops fetches /api/crops and renders a crop card for each listing."),
    (f"{S}::esc", "Asserts that esc escapes ampersand, less-than and greater-than characters."),
    (f"{S}::createCropCard", "Asserts that createCropCard builds call and WhatsApp links from the farmer phone number."),
    (f"{A}::PATCH /users/:id/approve", "Asserts that the approve route sets the user status to Approved and omits the password from the response."),
    ("server.js::USE /api/auth", "Asserts that server.js mounts authRoutes under /api/auth."),
    ("routes/userRoutes.js::DELETE /:id", "Asserts that the route deletes a user by id."),
    (f"{P}::GET /buyer/:email", "Asserts that the route returns completed transactions for the buyer email."),
    (f"{T}::statusBadge", "Asserts that statusBadge maps Pending, Approved and Rejected to CSS classes."),
    (f"{A}::DELETE /crops/:id", "Asserts that the route deletes a crop by id and responds with a removal message."),
]
for fqn, a in true_assert:
    cases.append(("faithful:true assertion", 0, fqn, claims([fqn], assertions=[a])))

# ---------- HALLUCINATED ----------
false_assert = [
    (f"{U}::POST /login", "Asserts that the login route hashes the password with bcrypt and returns a signed JWT token."),
    (f"{U}::POST /signup", "Asserts that signup sends a verification email and stores a hashed password."),
    (f"{A}::requireAdmin", "Asserts that requireAdmin validates a JWT bearer token from the Authorization header."),
    (f"{A}::GET /stats", "Asserts that the stats route caches the totals in Redis for one hour."),
    (f"{P}::POST /create-order", "Asserts that create-order uses Stripe to create a payment intent in USD."),
    (f"{P}::POST /verify-payment", "Asserts that verify-payment validates the webhook signature using HMAC."),
    (f"{P}::cashfreeRequest", "Asserts that cashfreeRequest retries failed requests three times with exponential backoff."),
    (f"{P}::GET /farmer/:identifier", "Asserts that the route returns pending payments for the farmer and requires admin access."),
    ("routes/cropRoutes.js::GET /", "Asserts that the route returns only crops from the logged in farmer and supports pagination."),
    ("routes/toolRoutes.js::GET /", "Asserts that the tools list is sorted by price and paginated with a limit of ten."),
    (f"{S}::loadCrops", "Asserts that loadCrops uses WebSocket to stream crops in real time."),
    (f"{S}::esc", "Asserts that esc encrypts the string before rendering it."),
    (f"{S}::createCropCard", "Asserts that createCropCard sends an SMS to the farmer using Twilio."),
    (f"{A}::PATCH /users/:id/approve", "Asserts that the approve route deletes the user and sends an approval email."),
    ("server.js::USE /api/auth", "Asserts that server.js mounts authRoutes under /api/v2/auth with rate limiting."),
    ("routes/userRoutes.js::DELETE /:id", "Asserts that the route soft deletes the user by setting a deletedAt timestamp."),
    (f"{P}::GET /buyer/:email", "Asserts that the route returns failed transactions for the buyer and charges a fee."),
    (f"{T}::statusBadge", "Asserts that statusBadge fetches the user status from the database."),
    (f"{A}::DELETE /crops/:id", "Asserts that the route archives the crop and notifies the farmer."),
    (f"{P}::POST /create-order", "Asserts that the order total includes GST calculated at eighteen percent."),
]
for fqn, a in false_assert:
    cases.append(("halluc:false assertion", 1, fqn, claims([fqn], assertions=[a])))

invented = ["sendOtpEmail", "hashPassword", "validateToken", "logAuditEvent", "refundOrder",
            "notifyFarmer", "cacheResult", "sendSms", "PaymentService.refund", "AuthService.authenticate",
            "UserController.register", "CropRepository.findAll", "verifyJwt", "RedisClient.get"]
route_chunks = [c for c in chunks if c["symbol_type"] in ("route", "function")]
for i in range(40):  # H1: invented symbol alongside real ones
    c = rng.choice(route_chunks)
    ph = rng.choice(invented)
    cases.append(("halluc:invented symbol (1 of 2 claimed)", 1, c["fqn"],
                  claims([c["fqn"], f"{c['file_path']}::{ph}"])))
for i in range(40):
    c = rng.choice(route_chunks)
    reals = [c["fqn"]] + [x["fqn"] for x in rng.sample(chunks, 3)]
    ph = rng.choice(invented)
    cases.append(("halluc:invented symbol (1 of 5 claimed)", 1, c["fqn"],
                  claims(reals + [ph])))
generic = ["login", "user", "crop", "admin", "routes", "handler", "get", "app", "payment", "order"]
for g in generic:  # H1b: invented but generic token (substring of real FQNs)
    c = rng.choice(route_chunks)
    cases.append(("halluc:invented generic-name symbol", 1, c["fqn"],
                  claims([c["fqn"], f"UnknownService.{g}"] if False else [c["fqn"], g + "Manager"])))
# H1c: names that are real substrings but not real symbols
for g in generic:
    c = rng.choice(route_chunks)
    cases.append(("halluc:symbol = substring of a real FQN", 1, c["fqn"], claims([c["fqn"], g])))

for c in chunks:  # H2: wrong parameters
    wrong = ["token", "userId", "payload"] if c["params"] == ["req", "res"] else ["req", "res"]
    cases.append(("halluc:wrong parameters", 1, c["fqn"],
                  claims([c["fqn"]], [{"symbol": c["fqn"], "claimed_params": wrong, "claimed_parent": c["parent"]}])))
for c in chunks:  # H3: wrong parent (only detectable if a parent is recorded)
    cases.append(("halluc:wrong parent", 1, c["fqn"],
                  claims([c["fqn"]], [{"symbol": c["fqn"], "claimed_params": c["params"], "claimed_parent": "WrongParent"}])))
inv_callees = ["sendOtpEmail", "hashPassword", "validateToken", "logAuditEvent", "refundOrder", "notifyFarmer", "cacheResult", "sendSms"]
for i in range(30):  # H4: edge to invented callee
    c = rng.choice(route_chunks)
    cases.append(("halluc:call edge to invented callee", 1, c["fqn"],
                  claims([c["fqn"]], edges=[{"caller": c["fqn"], "callee": rng.choice(inv_callees)}])))
real_calls = {(a, b) for a, b, _ in true_edges}
n = 0
while n < 30:  # H5: edge between two real symbols that do not call each other
    c1, c2 = rng.sample(chunks, 2)
    if (c1["fqn"], c2["fqn"]) in real_calls:
        continue
    cases.append(("halluc:call edge between real symbols (no such call)", 1, c1["fqn"],
                  claims([c1["fqn"]], edges=[{"caller": c1["fqn"], "callee": c2["fqn"]}])))
    n += 1

# ---------- RUN ----------
rows = []
for typ, label, fqn, cl in cases:
    r = engine.verify(cl, context_for(fqn))
    rows.append({"type": typ, "label": label, "decision": r["decision"], "score": r["trust_score"],
                 "layers": {k: v.get("score") for k, v in r.get("layers", {}).items()}})

def prf(rs, flag):
    tp = sum(1 for r in rs if r["label"] == 1 and flag(r))
    fp = sum(1 for r in rs if r["label"] == 0 and flag(r))
    fn = sum(1 for r in rs if r["label"] == 1 and not flag(r))
    tn = sum(1 for r in rs if r["label"] == 0 and not flag(r))
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(p, 3), recall=round(rc, 3), f1=round(f1, 3),
                accuracy=round((tp + tn) / len(rs), 3), fpr=round(fp / (fp + tn), 3) if fp + tn else 0.0)

summary = {
    "n_cases": len(rows), "n_faithful": sum(r["label"] == 0 for r in rows), "n_halluc": sum(r["label"] == 1 for r in rows),
    "flag_if_not_VERIFIED": prf(rows, lambda r: r["decision"] != "VERIFIED"),
    "flag_if_REJECTED_only": prf(rows, lambda r: r["decision"].startswith("REJECTED")),
}
per = collections.OrderedDict()
for typ in dict.fromkeys(r["type"] for r in rows):
    rs = [r for r in rows if r["type"] == typ]
    flagged = sum(r["decision"] != "VERIFIED" for r in rs)
    per[typ] = {"n": len(rs), "flagged": flagged, "rate": round(flagged / len(rs), 3),
                "mean_score": round(sum(r["score"] for r in rs) / len(rs), 3),
                "decisions": dict(collections.Counter(r["decision"] for r in rs))}
summary["per_type"] = per
print(json.dumps(summary, indent=1))
if out_path:
    json.dump({"summary": summary, "rows": rows}, open(out_path, "w"))
