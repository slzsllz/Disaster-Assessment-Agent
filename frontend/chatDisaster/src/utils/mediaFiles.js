const videoExtensions = new Set(['mp4', 'm4v', 'webm', 'ogv', 'ogg', 'mov', 'avi', 'mkv', 'mpeg', 'mpg', '3gp', 'ts', 'mts', 'm2ts'])

export function isVideoFile(file = {}) {
  const mime = String(file.type || file.mime_type || '').toLowerCase().split(';')[0].trim()
  const extension = String(file.name || '').split('.').pop().toLowerCase()
  return mime.startsWith('video/') || videoExtensions.has(extension)
}

export function videoPreviewUrl(url = '') {
  return /^\/api\/files\/[a-zA-Z0-9_-]+$/.test(url) ? `${url}/video-preview` : ''
}
