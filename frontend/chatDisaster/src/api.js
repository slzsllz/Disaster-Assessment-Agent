export async function apiFetch(path, options = {}, notifyUnauthorized = true) {
  const response = await fetch(path, { credentials: 'same-origin', ...options })
  if (response.status === 401 && notifyUnauthorized) {
    window.dispatchEvent(new Event('auth-expired'))
  }
  return response
}

export async function apiJson(path, options = {}, notifyUnauthorized = true) {
  const response = await apiFetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options.headers },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  }, notifyUnauthorized)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const error = new Error(typeof data.detail === 'string' ? data.detail : '请求失败，请检查输入后重试。')
    error.status = response.status
    throw error
  }
  return data
}
