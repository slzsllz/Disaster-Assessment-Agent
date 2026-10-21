<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { amapFeatures, fitMap, geometryPoints, pixelLocation, worldPixel } from '../utils/mapProvider.js'

const props = defineProps({ geometry: { type: [Object, String], default: null } })
const container = ref(null)
const size = ref([640, 640])
const viewport = ref({ center: [104.1954, 35.8617], zoom: 4 })
const displayedViewport = ref(null)
const imageUrl = ref('')
const loading = ref(false)
const error = ref('')
const features = computed(() => amapFeatures(props.geometry))
const allPoints = computed(() => features.value.flatMap(f => geometryPoints(f.geometry)))
const colors = { hospital: '#2563eb', shelter: '#16a34a', fire_station: '#ea580c' }
let observer, timer, controller, generation = 0, disposed = false, drag = null

function fit() {
  viewport.value = fitMap(allPoints.value, ...size.value)
  schedule()
}
function schedule() {
  clearTimeout(timer)
  // Prevent old responses overwriting a more recent drag, zoom or geometry change.
  generation++
  controller?.abort()
  timer = setTimeout(loadMap, 300)
}
async function loadMap() {
  const request = ++generation
  controller = new AbortController()
  const snapshot = { ...viewport.value, size: [...size.value], features: features.value }
  loading.value = true
  error.value = ''
  try {
    const query = new URLSearchParams({ longitude: snapshot.center[0].toFixed(6), latitude: snapshot.center[1].toFixed(6),
      zoom: snapshot.zoom, width: snapshot.size[0], height: snapshot.size[1] })
    const response = await fetch(`/api/maps/amap/static?${query}`, { signal: controller.signal })
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}))
      throw new Error(typeof detail.detail === 'string' ? detail.detail : '高德地图暂时不可用，请重试。')
    }
    const blob = await response.blob()
    if (!blob.type.startsWith('image/')) throw new Error('高德地图返回格式异常。')
    const url = URL.createObjectURL(blob)
    // Decode first, then change the basemap and overlays together.
    try {
      const img = new Image()
      img.src = url
      await img.decode()
    } catch {
      URL.revokeObjectURL(url)
      throw new Error('高德地图图片无法解析，请重试。')
    }
    if (disposed || request !== generation) { URL.revokeObjectURL(url); return }
    if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
    imageUrl.value = url
    displayedViewport.value = snapshot
  } catch (cause) {
    if (!disposed && request === generation && cause.name !== 'AbortError') error.value = cause.message
  } finally {
    if (!disposed && request === generation) loading.value = false
  }
}
function zoomBy(delta) {
  viewport.value = { ...viewport.value, zoom: Math.max(1, Math.min(17, viewport.value.zoom + delta)) }
  schedule()
}
function panStart(event) {
  if (!displayedViewport.value || event.button !== 0) return
  drag = { x: event.clientX, y: event.clientY, view: displayedViewport.value }
  event.currentTarget.setPointerCapture(event.pointerId)
}
function panEnd(event) {
  if (!drag) return
  const bounds = container.value.getBoundingClientRect(), view = drag.view
  const center = worldPixel(view.center, view.zoom)
  const dx = (event.clientX-drag.x)*view.size[0]/bounds.width, dy = (event.clientY-drag.y)*view.size[1]/bounds.height
  drag = null
  if (Math.abs(dx)+Math.abs(dy) < 4) return
  viewport.value = { center: pixelLocation([center[0]-dx, center[1]-dy], view.zoom), zoom: view.zoom }
  schedule()
}
const shapes = computed(() => {
  const view = displayedViewport.value
  if (!view) return []
  const center = worldPixel(view.center, view.zoom)
  const point = p => { const xy = worldPixel(p, view.zoom); return [xy[0]-center[0]+view.size[0]/2, xy[1]-center[1]+view.size[1]/2] }
  const path = rings => rings.map(ring => ring.map((p, i) => `${i ? 'L' : 'M'}${point(p).join(',')}`).join(' ')+' Z').join(' ')
  return view.features.flatMap((f, index) => {
    const { type, coordinates } = f.geometry, p = f.properties
    const base = { color: colors[p.destination_type] || '#ef4444', title: p.name || '灾害范围', id: index }
    if (type === 'Point') return [{ ...base, point: point(coordinates), incident: p.role === 'incident' }]
    if (type === 'LineString' || type === 'MultiLineString') return (type === 'LineString' ? [coordinates] : coordinates)
      .map((line, i) => ({ ...base, id: `${index}-${i}`, line: line.map(p => point(p).join(',')).join(' ') }))
    if (type === 'Polygon' || type === 'MultiPolygon') return (type === 'Polygon' ? [coordinates] : coordinates)
      .map((rings, i) => ({ ...base, id: `${index}-${i}`, polygon: path(rings) }))
    return []
  })
})
watch(() => props.geometry, fit)
onMounted(() => {
  observer = new ResizeObserver(entries => {
    const { width, height } = entries[0].contentRect
    if (width < 1 || height < 1) return
    // Keep the map aspect ratio when the drawer exceeds the static API's 1024px limit.
    const scale = Math.min(1, 1024/width, 1024/height)
    const next = [Math.max(64, Math.round(width*scale)), Math.max(64, Math.round(height*scale))]
    if (next[0] !== size.value[0] || next[1] !== size.value[1]) { size.value = next; fit() }
  })
  observer.observe(container.value)
  fit()
})
onUnmounted(() => {
  disposed = true
  generation++
  clearTimeout(timer)
  controller?.abort()
  observer?.disconnect()
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
})
</script>

<template>
  <div ref="container" class="amap-container" aria-label="高德地图">
    <div class="amap-canvas" @pointerdown="panStart" @pointerup="panEnd" @pointercancel="drag = null" @wheel.prevent="zoomBy($event.deltaY < 0 ? 1 : -1)">
      <img v-if="imageUrl" :src="imageUrl" alt="高德地图底图" draggable="false" />
      <svg v-if="imageUrl && displayedViewport" :viewBox="`0 0 ${displayedViewport.size.join(' ')}`" preserveAspectRatio="none" role="img" aria-label="灾害位置和救援路线">
        <g v-for="shape in shapes" :key="shape.id">
          <title>{{ shape.title }}</title>
          <polyline v-if="shape.line" :points="shape.line" fill="none" :stroke="shape.color" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
          <path v-else-if="shape.polygon" :d="shape.polygon" :fill="shape.color" fill-opacity="0.18" fill-rule="evenodd" :stroke="shape.color" stroke-width="2" />
          <template v-else-if="shape.point">
            <circle :cx="shape.point[0]" :cy="shape.point[1]" :r="shape.incident ? 8 : 6" :fill="shape.color" stroke="white" stroke-width="2" />
            <text :x="shape.point[0] + (shape.point[0] > displayedViewport.size[0]/2 ? -10 : 10)" :y="shape.point[1]-10"
              :text-anchor="shape.point[0] > displayedViewport.size[0]/2 ? 'end' : 'start'"
              paint-order="stroke" stroke="white" stroke-width="3" fill="#182230" font-size="12">{{ shape.title }}</text>
          </template>
        </g>
      </svg>
    </div>
    <div class="amap-controls" aria-label="高德地图操作">
      <button type="button" aria-label="放大地图" :disabled="viewport.zoom >= 17" @click="zoomBy(1)">＋</button>
      <button type="button" aria-label="缩小地图" :disabled="viewport.zoom <= 1" @click="zoomBy(-1)">−</button>
      <button type="button" @click="fit">定位全部</button>
    </div>
    <div v-if="error" class="amap-message error" role="alert">{{ error }} <button type="button" @click="schedule">重试</button></div>
    <div v-else-if="loading" class="amap-message" role="status">高德地图加载中...</div>
    <div class="amap-attribution">高德地图 · 拖动平移 / 滚轮缩放</div>
  </div>
</template>

<style scoped>
.amap-container { position: relative; flex: 1 1 auto; width: 100%; min-height: 0; overflow: hidden; border: 1px solid #e2e5eb; border-radius: 8px; background: #edf1f4; }
.amap-canvas { position: absolute; inset: 0; touch-action: none; cursor: grab; }
.amap-canvas:active { cursor: grabbing; }
.amap-canvas img, .amap-canvas svg { position: absolute; inset: 0; width: 100%; height: 100%; user-select: none; }
.amap-controls { position: absolute; right: 12px; top: 12px; display: flex; flex-direction: column; gap: 4px; }
.amap-controls button, .amap-message button { background: white; color: #24334a; border: 1px solid #cdd5df; border-radius: 5px; padding: 8px 10px; cursor: pointer; }
.amap-controls button:disabled { opacity: .5; cursor: default; }
.amap-message { position: absolute; top: 12px; left: 12px; right: 110px; padding: 10px; border-radius: 6px; background: #ffffffed; font-size: 13px; color: #334155; }
.amap-message.error { color: #b42318; }
.amap-attribution { position: absolute; bottom: 0; right: 0; padding: 3px 7px; font-size: 11px; color: #475569; background: #ffffffdb; pointer-events: none; }
</style>
