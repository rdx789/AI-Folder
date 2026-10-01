# Adapted S2 read-tool matrix

Derived from the Lesson 10 classroom S2; document searches run through the injected retriever. Turn 10 produces a typed handoff. Every intent alternative is checked by `python -m maya.matrix`.

| Turn | Accepted intent | Required operational reads | Retrieval | Typed handoff |
| --- | --- | --- | --- | --- |
| 1 | employee_lookup | get_employee | no | no |
| 2 | onboarding_status | none | yes | no |
| 2 | employee_lookup | none | yes | no |
| 3 | onboarding_status | list_onboarding_tasks | no | no |
| 4 | ticket_status | list_employee_tickets | no | no |
| 5 | equipment_request | check_asset_inventory | no | no |
| 5 | onboarding_status | check_asset_inventory | no | no |
| 6 | equipment_request | get_policy | no | no |
| 6 | policy_question | get_policy | no | no |
| 7 | recall | none | no | no |
| 8 | subscription_review | check_software_subscription | no | no |
| 8 | access_request | check_software_subscription | no | no |
| 9 | subscription_review | none | yes | no |
| 9 | policy_question | none | yes | no |
| 10 | access_request | none | no | yes |
| 11 | onboarding_status | list_onboarding_tasks | no | no |
| 12 | recall | none | no | no |

All listed reads are reachable from their selected loadouts. Unknown-intent and rearm fallbacks contain only the seven authorized reads. `create_access_request` is absent everywhere. Reachability does not prove live intent classification or correct tool use.
