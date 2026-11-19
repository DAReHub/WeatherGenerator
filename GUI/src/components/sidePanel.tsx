import { Separator } from "@/components/ui/separator"
import { Label } from "@/components/ui/label"
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox"
import type { UkcpModel, UkcpModelGroup } from "@/types/ukcp"
import * as React from "react"
import { GenerateRealsDialog } from "@/components/generateRealsDialog.tsx"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"

type SidePanelProps = {
  PERIODS: { label: string; abbr: string }[]
  variables: string[]
  setVariables: React.Dispatch<React.SetStateAction<string[]>>
  UKCP_MODELS: UkcpModelGroup[]
  model: string
  setModel: (model: string) => void
  period: number
  setPeriod: (period: string) => void
  easting: number | null
  northing: number | null
  BASEMAPS: { label: string; value: string }[]
  basemap: string
  setBasemap: (basemap: string) => void
}

export function SidePanel({
                            PERIODS,
                            variables,
                            setVariables,
                            UKCP_MODELS,
                            model,
                            setModel,
                            period,
                            setPeriod,
                            BASEMAPS,
                            basemap,
                            setBasemap,
                            easting,
                            northing,
                          }: SidePanelProps) {
  const dataSection = (
    <div className="flex flex-col gap-3">
      <div className="py-3">
        <GenerateRealsDialog
          easting={easting}
          northing={northing}
          UKCP_MODELS={UKCP_MODELS}
          model={model}
          setModel={setModel}
          variables={variables}
          setVariables={setVariables}
          period={period}
          setPeriod={setPeriod}
          PERIODS={PERIODS}
        />
      </div>
    </div>
  )

  const displaySection = (
    <Card className="border-dashed bg-muted/40 shadow-none">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-muted-foreground">
          Display Settings
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-3">
          <Label htmlFor="combobox-basemap">Basemap</Label>
          <Combobox
            items={BASEMAPS}
            value={BASEMAPS.find((b) => b.value === basemap)}
            onValueChange={(item) => setBasemap(item.value)}
          >
            <ComboboxInput placeholder="Select a basemap" />
            <ComboboxContent>
              <ComboboxEmpty>No items found.</ComboboxEmpty>
              <ComboboxList>
                {(item) => (
                  <ComboboxItem key={item.value} value={item}>
                    {item.label}
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </div>
      </CardContent>
    </Card>
  )

  return (
    <div className="flex flex-col gap-6 p-6">
      {dataSection}
      {displaySection}
    </div>
  )
}