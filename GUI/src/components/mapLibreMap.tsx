import { useEffect, useRef, useState } from "react"
import maplibregl from "maplibre-gl"
import "maplibre-gl/dist/maplibre-gl.css"
import { useTheme } from "@/components/theme-provider"
import {
  ChevronUp,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Plus,
  Minus,
  Loader2,
} from "lucide-react"
import { Button } from "@/components/ui/button"

const DEFAULT_VIEW: { center: [number, number]; zoom: number } = {
  center: [-4, 54.8],
  zoom: 5.5,
}

const MAPTILER_KEY = import.meta.env.VITE_MAPTILER_KEY

const PAN_STEP_PX = 100

function resolveStyle(basemap: string, theme: string): string {
  const id =
    basemap === "theme"
      ? theme === "dark"
        ? "backdrop-v4-dark"
        : "backdrop-v4-light"
      : basemap
  return `https://api.maptiler.com/maps/${id}/style.json?key=${MAPTILER_KEY}`
}

type MapLibreMapProps = {
  easting: number | null
  setEasting: (easting: number | null) => void
  northing: number | null
  setNorthing: (northing: number | null) => void
  basemap: string
}

export function MapLibreMap({
  easting,
  setEasting,
  northing,
  setNorthing,
  basemap,
}: MapLibreMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)
  const { theme } = useTheme()

  // True until the map has finished its very first load (style + grid
  // source). We intentionally never flip this back to true on later
  // basemap/theme changes — that's a background swap, not the initial load.
  const [isLoading, setIsLoading] = useState(true)
  const hasLoadedOnceRef = useRef(false)

  // Tracks the currently-selected cell's grid indices (not just the
  // easting/northing passed up to React) so the highlight can be
  // reapplied whenever the grid layer is recreated (e.g. after a
  // basemap/style change wipes all sources & layers).
  const selectedIndexRef = useRef<{ x: unknown; y: unknown } | null>(null)

  const applyHighlight = () => {
    if (!map.current || !map.current.getLayer("grid-highlight")) return

    const selected = selectedIndexRef.current
    map.current.setFilter(
      "grid-highlight",
      selected
        ? [
            "all",
            ["==", ["get", "HUK_x_index"], selected.x],
            ["==", ["get", "HUK_y_index"], selected.y],
          ]
        : ["==", ["id"], ""]
    )
  }

  const addGridLayer = () => {
    if (!map.current) return

    if (!map.current.getSource("grid")) {
      map.current.addSource("grid", {
        type: "geojson",
        data: "/HUKMETA2_4326.geojson",
      })

      map.current.addLayer({
        id: "grid-fill",
        type: "fill",
        source: "grid",
        paint: {
          "fill-color": "#000",
          "fill-opacity": 0.05,
        },
      })

      map.current.addLayer({
        id: "grid-outline",
        type: "line",
        source: "grid",
        paint: {
          "line-color": "rgba(255,0,0,0.2)",
          "line-width": 3,
        },
      })

      map.current.addLayer({
        id: "grid-highlight",
        type: "line",
        source: "grid",
        paint: {
          "line-color": "#00ffd9",
          "line-width": 5,
        },
        filter: ["==", ["id"], ""],
      })
    }

    // Restore any prior selection — needed because setStyle() removes
    // every source/layer, so a freshly (re)created grid-highlight layer
    // always starts with an empty filter regardless of past state.
    applyHighlight()
  }

  // Initialise map
  useEffect(() => {
    if (map.current || !mapContainer.current) return

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: resolveStyle(basemap, theme),
      center: DEFAULT_VIEW.center,
      zoom: DEFAULT_VIEW.zoom,
    })
    currentStyleUrlRef.current = resolveStyle(basemap, theme)

    map.current.on("load", () => {
      addGridLayer()

      // Plain click = select single cell
      map.current!.on("click", "grid-fill", (e) => {
        const feature = e.features?.[0]
        if (!feature) return

        const x = feature.properties?.HUK_x_index
        const y = feature.properties?.HUK_y_index

        selectedIndexRef.current = { x, y }

        setEasting(feature.properties?.easting ?? null)
        setNorthing(feature.properties?.northing ?? null)

        applyHighlight()
      })

      map.current!.on("mouseenter", "grid-fill", () => {
        if (map.current) map.current.getCanvas().style.cursor = "pointer"
      })

      map.current!.on("mouseleave", "grid-fill", () => {
        if (map.current) map.current.getCanvas().style.cursor = ""
      })
    })

    // Fires once the style, all sources (including the grid geojson),
    // and all tiles have finished loading — the natural "fully ready"
    // signal for the initial load.
    map.current.once("idle", () => {
      hasLoadedOnceRef.current = true
      setIsLoading(false)
    })

    return () => {
      map.current?.remove()
      map.current = null
    }
  }, [])

  const currentStyleUrlRef = useRef<string>("")

  // Theme / basemap changes — only touches the map style when the
  // resolved URL actually changes, and never on unrelated re-renders.
  useEffect(() => {
    if (!map.current) return

    const nextUrl = resolveStyle(basemap, theme)
    if (nextUrl === currentStyleUrlRef.current) return
    currentStyleUrlRef.current = nextUrl

    map.current.once("style.load", () => {
      addGridLayer()
    })
    map.current.setStyle(nextUrl)
  }, [theme, basemap])

  const zoomIn = () => map.current?.zoomIn()
  const zoomOut = () => map.current?.zoomOut()
  const panBy = (dx: number, dy: number) =>
    map.current?.panBy([dx, dy], { duration: 300 })

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div ref={mapContainer} style={{ width: "100%", height: "100%" }} />

      {isLoading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-sidebar">
          <div className="flex flex-col items-center gap-2">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Loading map…</span>
          </div>
        </div>
      )}

      {/* Zoom controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-1">
        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={zoomIn}
          aria-label="Zoom in"
        >
          <Plus className="h-4 w-4" />
        </Button>

        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={zoomOut}
          aria-label="Zoom out"
        >
          <Minus className="h-4 w-4" />
        </Button>
      </div>

      {/* Pan controls */}
      <div className="absolute top-4 right-20 grid grid-cols-3 gap-1">
        <div />

        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={() => panBy(0, -PAN_STEP_PX)}
          aria-label="Pan up"
        >
          <ChevronUp className="h-4 w-4" />
        </Button>

        <div />

        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={() => panBy(-PAN_STEP_PX, 0)}
          aria-label="Pan left"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>

        <div />

        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={() => panBy(PAN_STEP_PX, 0)}
          aria-label="Pan right"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>

        <div />

        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={() => panBy(0, PAN_STEP_PX)}
          aria-label="Pan down"
        >
          <ChevronDown className="h-4 w-4" />
        </Button>

        <div />
      </div>
    </div>
  )
}
