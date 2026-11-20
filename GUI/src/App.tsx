import { MapLibreMap } from "./components/mapLibreMap.tsx"
import { ThemeProvider } from "@/components/theme-provider"
import { useState } from "react"
import { AboutDialog } from "@/components/aboutDialog.tsx"
import { SidePanel } from "@/components/sidePanel.tsx"
import ukcpModelsData from "@/app-data/ukcp-models.json"
import type { UkcpModel, UkcpModelGroup } from "@/types/ukcp"
import { Header } from "@/components/layout/header.tsx"
import { Card } from "@/components/ui/card.tsx"

const UKCP_MODELS = ukcpModelsData as UkcpModelGroup[]

const PERIODS = [
  { label: "2041-2060", abbr: 2041 },
  { label: "2061-2080", abbr: 2061 },
]

const BASEMAPS = [
  { label: "Hillshade (light/dark)", value: "theme" },
  { label: "Satellite", value: "hybrid-v4" },
  { label: "Transport", value: "streets-v4" },
]

function App() {
  // initial state
  const [aboutOpen, setAboutOpen] = useState(true)
  const [variables, setVariables] = useState<string[]>(["Rainfall"])
  const [model, setModel] = useState("M01")
  const [easting, setEasting] = useState<number | null>(null)
  const [northing, setNorthing] = useState<number | null>(null)
  const [basemap, setBasemap] = useState("theme")
  const [period, setPeriod] = useState(2041)

  return (
    <ThemeProvider defaultTheme="light" storageKey="vite-ui-theme">
      <div className="flex h-screen w-screen flex-col gap-4 p-4 lg:flex-row">
        <div className="flex max-h-[45vh] w-full flex-col gap-4 overflow-x-visible overflow-y-auto px-1 lg:max-h-none lg:w-full lg:max-w-2xl">
          <Header onAboutClick={() => setAboutOpen(true)} />
          <SidePanel
            PERIODS={PERIODS}
            variables={variables}
            setVariables={setVariables}
            UKCP_MODELS={UKCP_MODELS}
            model={model}
            setModel={setModel}
            period={period}
            setPeriod={setPeriod}
            easting={easting}
            northing={northing}
            BASEMAPS={BASEMAPS}
            basemap={basemap}
            setBasemap={setBasemap}
          />
        </div>

        <div className="relative min-h-0 flex-1">
          <Card className="h-full w-full rounded-2xl border bg-sidebar p-0 shadow-[-6px_8px_16px_rgba(0,0,0,0.12)] backdrop-blur-sm">
            <div className="h-full w-full overflow-hidden rounded-2xl">
              <MapLibreMap
                easting={easting}
                setEasting={setEasting}
                northing={northing}
                setNorthing={setNorthing}
                basemap={basemap}
              />
            </div>
          </Card>
        </div>

        <AboutDialog aboutOpen={aboutOpen} setAboutOpen={setAboutOpen} />
      </div>
    </ThemeProvider>
  )
}

export default App
