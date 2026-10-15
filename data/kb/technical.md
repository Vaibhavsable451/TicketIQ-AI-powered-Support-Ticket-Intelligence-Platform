# Technical troubleshooting
For crashes, clear the app cache, update to the latest version and retry. If the crash persists, collect logs from Help > Diagnostics and attach them to the ticket.
Sync errors (codes 400-599) usually resolve after signing out and in again. Platform incidents are published on the status page.
API 500 errors: check rate limits, retry with exponential backoff, and include the request ID from the response header when contacting support. Slow dashboards are often caused by large date ranges; narrow the filter.