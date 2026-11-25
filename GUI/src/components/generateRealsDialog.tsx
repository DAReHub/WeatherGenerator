import { Field, FieldGroup, FieldSeparator } from "@/components/ui/field"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs.tsx"
import { useState, useRef } from "react"
import { useDownloadReals } from "@/hooks/UseDownloadReals"
import { Loader2 } from "lucide-react"
import type { UkcpModel, UkcpModelGroup } from "@/types/ukcp"
import * as React from "react"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectItem,
  SelectContent,
  SelectSeparator,
  SelectGroup,
  SelectLabel,
} from "@/components/ui/select.tsx"
import { useEffect } from "react"
import { cn } from "@/lib/utils"
import {
  MapPin,
  CloudRain,
  ThermometerSun,
  MonitorCog,
  AlertTriangle,
  Wind,
  WavesVertical,
  Sun,
  Gauge,
  Dices
} from "lucide-react"
import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

function UkcpModelPicker({
  UKCP_MODELS,
  model,
  setModel,
}: {
  UKCP_MODELS: UkcpModelGroup[]
  model: string
  setModel: (model: string) => void
}) {
  const allModels = React.useMemo(
    () => UKCP_MODELS.flatMap((g) => g.items),
    [UKCP_MODELS]
  )

  const selected = allModels.find((m) => m.name === model)

  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
      <div className="w-full sm:w-48 sm:shrink-0">
        <Select value={model} onValueChange={setModel}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select a UKCP18 Ensemble Member" />
          </SelectTrigger>

          <SelectContent>
            {UKCP_MODELS.map((group, index) => (
              <React.Fragment key={group.set}>
                {index > 0 && <SelectSeparator />}
                <SelectGroup>
                  <SelectLabel>{group.set}</SelectLabel>
                  {group.items.map((item) => (
                    <SelectItem key={item.name} value={item.name}>
                      {item.name}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </React.Fragment>
            ))}
          </SelectContent>
        </Select>
      </div>

      {selected && (
        <div className="flex-1 border-l border-border pl-4 text-sm text-muted-foreground">
          {selected.description}
        </div>
      )}
    </div>
  )
}

function RealisationsForm({
  spatial_res,
  temporal_res,
  period,
  model,
  easting,
  northing,
  real_count,
  disabled,
  variables,
  seed,
}: {
  spatial_res: number
  temporal_res: string
  period: number
  model: string
  easting: number | null
  northing: number | null
  real_count: number
  disabled: boolean
  variables: string[]
  seed: number
}) {
  const { downloadReals, isLoading, error } = useDownloadReals()

  const handleSubmit = () => {
    if (easting === null || northing === null) return

    downloadReals({
      ensemble_member: model,
      start_year: period,
      period: 20,
      resolution: temporal_res,
      grid_size_km: spatial_res,
      coordinates: [{ easting, northing }],
      realisation_count: real_count,
      variables: variables,
      seed: seed,

      // variable: variables,
      // spatial_res: spatial_res,
      // temporal_res: temporal_res,
      // period: period,
      // model: model,
      // locations: [{ easting, northing }],
      // real_count: real_count,
    })
  }

  return (
    <div className="w-full">
      <Button
        onClick={handleSubmit}
        disabled={isLoading || disabled}
        variant={error ? "destructive" : "default"}
        className="h-11 w-full rounded-lg from-sky-600 font-medium shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md active:translate-y-0 disabled:hover:translate-y-0"
      >
        {isLoading ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Generating...
          </>
        ) : (
          <>
            <MonitorCog className="h-4 w-4" />
            Generate!
          </>
        )}
      </Button>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  )
}

function NumberInput({
  value,
  onValueChange,
  invalid,
  max = 100,
}: {
  value: string
  onValueChange: (value: string) => void
  invalid?: boolean
  max?: number
}) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value

    if (raw === "") {
      onValueChange("")
      return
    }

    // only digits allowed
    if (!/^\d+$/.test(raw)) return

    // reject leading zeros entirely ("0", "00", "05", etc.)
    // no valid value in [1, max] starts with 0
    if (/^0/.test(raw)) return

    const num = Number(raw)

    if (num > max) return

    onValueChange(raw)
  }

  const handleBlur = () => {
    if (value === "") return
    const num = Number(value)
    if (num < 1) onValueChange("1")
    if (num > max) onValueChange(String(max))
  }

  return (
    <Input
      type="text"
      inputMode="numeric"
      pattern="[0-9]*"
      value={value}
      onChange={handleChange}
      onBlur={handleBlur}
      placeholder={`1-${max}`}
      className={invalid ? "border-red-500 ring-1 ring-red-500" : undefined}
    />
  )
}

export function GenerateRealsDialog({
  UKCP_MODELS,
  model,
  setModel,
  variables,
  setVariables,
  period,
  setPeriod,
  PERIODS,
  easting,
  northing,
}: {
  UKCP_MODELS: UkcpModelGroup[]
  model: string
  setModel: (model: string) => void
  variables: string[]
  setVariables: React.Dispatch<React.SetStateAction<string[]>>
  period: number
  setPeriod: (period: string) => void
  PERIODS: { label: string; abbr: string }[]
  easting: number | null
  northing: number | null
}) {
  const dialogContentRef = useRef<HTMLDivElement>(null)

  const [realCount, setRealCount] = useState<string>("100")
  const [temporalResolution, setTemporalResolution] = useState<string>("HOURLY")

  const maxReals = Number(import.meta.env.VITE_MAX_REALS)

  const handleSelectVariable = (id: string, checked: boolean) => {
    setVariables((prev) => {
      if (checked) {
        const deps: readonly string[] =
          VARIABLES.find((v) => v.id === id)?.dependencies ?? []
        return Array.from(new Set([...prev, id, ...deps]))
      }
      // unticking: remove this variable and anything that depends on it
      const removed = new Set<string>([
        id,
        ...VARIABLES.filter((v) =>
          (v.dependencies as readonly string[]).includes(id)
        ).map((v) => v.id),
      ])
      return prev.filter((v) => !removed.has(v))
    })
  }

  const forcesDaily = variables.length > 1

  const hasLocation = easting !== null && northing !== null

  const allFieldsFilled =
    hasLocation &&
    model !== "" &&
    variables.length > 0 &&
    period !== "" &&
    realCount !== "" &&
    temporalResolution !== ""

  const invalidClass = "border-red-500 ring-1 ring-red-500"

  useEffect(() => {
    if (forcesDaily && temporalResolution !== "DAILY") {
      setTemporalResolution("DAILY")
    }
  }, [forcesDaily, temporalResolution])

  const VARIABLES = [
    {
      id: "Rainfall",
      label: "Rainfall",
      icon: CloudRain,
      locked: true,
      resolution: "Hourly, Daily",
      units: "mm",
      dependencies: []
    },
    {
      id: "Temperature",
      label: "Temperature",
      icon: ThermometerSun,
      locked: false,
      resolution: "Daily",
      units: "°C",
      dependencies: ["Rainfall"]
    },
    {
      id: "Wind Speed",
      label: "Wind Speed",
      icon: Wind,
      locked: false,
      resolution: "Daily",
      units: "m/s",
      dependencies: ["Rainfall", "Temperature"]
    },
    {
      id: "Vapour Pressure",
      label: "Vapour Pressure",
      icon: Gauge,
      locked: false,
      resolution: "Daily",
      units: "kPa",
      dependencies: ["Rainfall", "Temperature"]
    },
    {
      id: "Sunshine Duration",
      label: "Sunshine Duration",
      icon: Sun,
      locked: false,
      resolution: "Daily",
      units: "hours",
      dependencies: ["Rainfall", "Temperature"]
    },
    {
      id: "Potential Evapotranspiration",
      label: "Potential Evapotranspiration",
      icon: WavesVertical,
      locked: false,
      resolution: "Daily",
      units: "mm/d",
      dependencies: ["Rainfall", "Temperature", "Wind Speed", "Vapour Pressure", "Sunshine Duration"]
    }
  ] as const

  const randomSeed = () => String(Math.floor(Math.random() * 1_000_000) + 1)
  const [seed, setSeed] = useState(randomSeed)

  return (
    <form>
      <FieldGroup>
        {/* Location */}
        <Field>
          <Label htmlFor="selected-location">Location</Label>
          <div
            id="selected-location"
            className={cn(
              "flex items-center gap-2 rounded-md border px-3 py-2 text-sm",
              hasLocation
                ? "border-border bg-muted/40 text-foreground"
                : `${invalidClass} text-destructive`
            )}
          >
            <MapPin className="h-4 w-4 shrink-0 text-muted-foreground" />
            {hasLocation ? (
              <span>
                Easting <span className="font-medium">{easting}</span>
                {", "}
                Northing <span className="font-medium">{northing}</span>
              </span>
            ) : (
              <span>Click a cell on the map to select a location</span>
            )}
          </div>
        </Field>

        <FieldSeparator />

        {/* Model */}
        <Field>
          <Label htmlFor="combobox-model">UKCP18 Ensemble Member</Label>
          <UkcpModelPicker
            UKCP_MODELS={UKCP_MODELS}
            model={model}
            setModel={setModel}
          />
        </Field>

        <FieldSeparator />

        {/* Scenario */}
        <Field>
          {/*<Label>Climate Variables</Label>*/}
          <div className="overflow-hidden rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8" />
                  <TableHead>Climate Variables</TableHead>
                  <TableHead>Temporal Resolution</TableHead>
                  <TableHead>Measurement Units</TableHead>
                  <TableHead>Dependent Variables</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {VARIABLES.map(
                  ({ id, label, icon: Icon, locked, resolution, units, dependencies }) => {
                    const checked = variables.includes(id)
                    return (
                      <TableRow
                        key={id}
                        data-state={checked ? "selected" : undefined}
                        className={cn(!locked && "cursor-pointer")}
                        onClick={() => !locked && handleSelectVariable(id, !checked)}
                      >
                        <TableCell>
                          <Checkbox
                            id={`variable-${id}`}
                            name={`variable-${id}`}
                            checked={checked}
                            disabled={locked}
                            onClick={(e) => e.stopPropagation()}
                            onCheckedChange={(c) => handleSelectVariable(id, c === true)}
                          />
                        </TableCell>
                        <TableCell className="font-medium">
                          <Label
                            htmlFor={`variable-${id}`}
                            className="flex items-center gap-2"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <Icon className="h-4 w-4" />
                            {label}
                            {locked && (
                              <span className="text-xs font-normal text-muted-foreground">
                                (required)
                              </span>
                            )}
                          </Label>
                        </TableCell>
                        <TableCell>{resolution}</TableCell>
                        <TableCell className="text-muted-foreground">{units}</TableCell>
                        <TableCell>
                          {dependencies.length === 0 ? (
                            <span className="text-xs text-muted-foreground">None</span>
                          ) : (
                            <div className="flex items-center gap-1.5 text-muted-foreground">
                              {dependencies.map((dep) => {
                                const DepIcon = VARIABLES.find((v) => v.id === dep)?.icon
                                return DepIcon ? (
                                  <span key={dep} title={dep}>
                                    <DepIcon className="h-4 w-4" />
                                  </span>
                                ) : null
                              })}
                            </div>
                          )}
                        </TableCell>
                      </TableRow>
                    )
                  }
                )}
              </TableBody>
            </Table>
          </div>
        </Field>

        <FieldSeparator />

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Field>
            <Label htmlFor="radio-period">Time Period</Label>
            <Tabs value={period} onValueChange={setPeriod} className="w-full">
              <TabsList className={cn("w-full", period === "" && invalidClass)}>
                {PERIODS.map((p) => (
                  <TabsTrigger key={p.abbr} value={p.abbr} className="flex-1">
                    {p.label}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </Field>

          <Field>
            <Label>Temporal Resolution</Label>
            <Tabs
              value={temporalResolution}
              onValueChange={setTemporalResolution}
              className="w-full"
            >
              <TabsList
                className={cn("w-full", temporalResolution === "" && invalidClass)}
              >
                <TabsTrigger value="HOURLY" disabled={forcesDaily} className="flex-1">
                  Hourly
                </TabsTrigger>
                <TabsTrigger value="DAILY" className="flex-1">
                  Daily
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </Field>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Field>
            <Label>Realisations</Label>
            <NumberInput
              value={realCount}
              onValueChange={setRealCount}
              invalid={realCount === ""}
              max={maxReals}
            />
          </Field>

          <Field>
            <Label htmlFor="input-seed">Seed</Label>
            <div className="flex gap-2">
              <Input
                id="input-seed"
                type="text"
                inputMode="numeric"
                pattern="[0-9]*"
                value={seed}
                onChange={(e) => {
                  if (/^\d*$/.test(e.target.value)) setSeed(e.target.value)
                }}
              />
              <Button type="button" onClick={() => setSeed(randomSeed())}>
                <Dices/>
              </Button>
            </div>
          </Field>
        </div>

        <FieldSeparator />

        <Field>
          <RealisationsForm
            spatial_res={5}
            temporal_res={temporalResolution}
            period={period}
            model={model}
            easting={easting}
            northing={northing}
            real_count={realCount === "" ? 0 : Number(realCount)}
            disabled={!allFieldsFilled}
            variables={variables}
            seed={seed === "" ? 1 : Number(seed)}
          />
        </Field>
      </FieldGroup>
    </form>
  )
}
