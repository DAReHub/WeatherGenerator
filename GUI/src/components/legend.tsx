import { useEffect, useState } from "react"

type LegendProps = {
  selectedColour: string
  label: string
  unit: string
  min: number
  max: number
  numStops?: number
  reverseColourScale: boolean
}

export function Legend({
  selectedColour,
  label,
  unit,
  min,
  max,
  numStops = 5,
  reverseColourScale,
}: LegendProps) {
  const [gradient, setGradient] = useState<string>("")

  const stops = Array.from({ length: numStops }, (_, i) =>
    Math.round(min + (i / (numStops - 1)) * (max - min))
  )

  useEffect(() => {
    fetch(
      `http://localhost:8000/colorMaps/${
        reverseColourScale ? `${selectedColour}_r` : selectedColour
      }?f=json`
    )
      .then((res) => res.json())
      .then((data) => {
        // Sample ~10 evenly spaced colours from the 256 entries
        const entries = Object.entries(data) as [string, number[]][]
        const step = Math.floor(entries.length / 10)
        const sampled = entries.filter((_, i) => i % step === 0)

        const colourStops = sampled
          .map(([, [r, g, b, a]]) => `rgba(${r},${g},${b},${a / 255})`)
          .join(", ")

        setGradient(`linear-gradient(to right, ${colourStops})`)
      })
      .catch((err) => console.error("Failed to fetch colormap:", err))
  }, [selectedColour, reverseColourScale])

  return (
    <div className="absolute top-4 right-4 z-50 rounded-xl border bg-background/80 px-3 py-2 shadow-lg backdrop-blur-sm">
      <p className="mb-1.5 text-xs font-medium">
        {label} ({unit})
      </p>
      <div
        className="h-3 w-80 rounded-sm"
        style={{ background: gradient || "transparent" }}
      />
      <div className="mt-1 flex w-80 justify-between">
        {stops.map((v) => (
          <span key={v} className="text-xs text-muted-foreground">
            {v}
          </span>
        ))}
      </div>
    </div>
  )
}
