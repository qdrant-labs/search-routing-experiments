// One thin fetch wrapper: JSON in/out, errors surfaced with the server's
// `detail` so a 409/422 reads as a sentence, not a status code.
async function request(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(body?.detail || `${res.status} ${res.statusText}`);
  }
  return body;
}

export const get = (path) => request(path);

export const post = (path, body) =>
  request(path, {
    method: "POST",
    headers: body ? { "content-type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
