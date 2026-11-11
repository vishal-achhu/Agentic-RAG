# Nimbus API Rate Limits

## Default limits
The REST API allows 600 requests per minute per workspace on the Starter plan and 3,000 requests per minute on the Pro plan. Enterprise limits are negotiated per contract.

## Handling throttling
When the limit is exceeded the API returns HTTP 429 with a Retry-After header. Clients should use exponential backoff with jitter. Bulk export endpoints are limited to 10 concurrent jobs per workspace.

## Pagination
List endpoints return at most 200 items per page. Use the cursor value in the response to fetch the next page.

## Authentication
API requests are authenticated with bearer tokens. Tokens expire after 90 days and can be rotated from the Developer settings page.
