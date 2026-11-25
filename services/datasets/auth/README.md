# P2-12 access control (`services.datasets.auth`)

Default-deny authorisation for dataset lineage and deletion workflows. This
package does **not** contain an identity provider, login flow or role-assignment
API. A production adapter must verify credentials out of band and construct a
`services.datasets.lineage.Principal` with one verified `Role` enum member.

## Public API

```python
from services.datasets.auth import Authorizer, AuthorizationError
from services.datasets.lineage import (
    AuditPurpose,
    Capability,
    Principal,
    Resource,
    Role,
)

authorizer = Authorizer(audit_sink, clock)
decision = authorizer.check(
    Principal("admin-1", Role.ADMIN),
    Capability.DELETION_AUTHORIZE,
    Resource.DELETION,
    purpose=AuditPurpose.GOVERNANCE,
)
```

- `check()` returns an `AuthorizationDecision` and always audits the outcome.
- `require()` calls `check()` and raises `AuthorizationError` on denial.
- `assert_pseudonymous_call_ref()` rejects raw `call_id` values passed as audit
  metadata; only `call_sha256:<32 hex>` references are accepted.

## Role matrix is immutable

The matrix lives in `services.datasets.lineage.roles` as a module-local
`MappingProxyType` over copied `frozenset` values. It is deliberately **not**
re-exported from `services.datasets.lineage` and is not part of the documented
public API. `capabilities_for(role)` is the only supported accessor.

- Assigning to a role's capability set raises `TypeError`.
- Mutating a returned frozenset raises `AttributeError`.
- Strings such as `"admin"` cannot become roles; `Principal` requires a `Role`
  enum member.
- There is no endpoint or method that changes a principal's role. Reviewers and
  ML engineers cannot authorise deletion or change consent; attempts are
  audited as denials and perform no side effects.

## Fixed audit codes

Every decision produces one `AuditEvent` with fixed enum codes:

| Field | Accepted values |
|---|---|
| `action` | `Capability` values plus `deletion_execute` and `unknown_action` (`AuditAction`) |
| `resource` | `Resource` members plus `unknown_resource` |
| `purpose` | `AuditPurpose` members plus `unknown_purpose` |
| `reason` | `AuditReason` members such as `capability_granted`, `capability_missing`, `unknown_action`, `unknown_resource`, `unknown_purpose`, `unknown_call`, `sink_failure` and deletion reason codes |

Invalid action/resource/purpose inputs are handled before any `.value` access:
a `DENY` audit event with the matching `unknown_*` code is recorded, then an
`AuthorizationError` is raised. Raw input strings are never copied into the
audit event or its hash. `InMemoryAuditSink.record()` independently rejects
events whose codes are not enum members.

## Verification

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/lineage_access -q
.\.venv\Scripts\python.exe -m ruff check services/datasets/auth services/datasets/lineage tests/unit/lineage_access
.\.venv\Scripts\python.exe -m ruff format --check services/datasets/auth services/datasets/lineage tests/unit/lineage_access
.\.venv\Scripts\python.exe -m mypy services/datasets/auth services/datasets/lineage tests/unit/lineage_access
```

No secrets, real guest identifiers or free-form actor text are accepted by
this package. See `services/datasets/lineage/README.md` for the injected sink
and clock seams used by tests.
