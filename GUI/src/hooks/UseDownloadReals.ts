import { useState, useCallback } from "react"

interface Coordinates {
  easting: number
  northing: number
}

interface RealsParams {
  ensemble_member: string
  start_year: number
  period: number
  resolution: string
  grid_size_km: number
  coordinates: Coordinates[]
  realisation_count: number
  variables: []
  seed: number
}

interface UseDownloadRealsResult {
  downloadReals: (params: {
    ensemble_member: string;
    start_year: number;
    period: number;
    resolution: string;
    grid_size_km: number;
    coordinates: { easting: number; northing: number }[];
    realisation_count: number;
    variables: string[];
    seed: number
  }) => Promise<void>
  isLoading: boolean
  error: string | null
}

/**
 * Hook to call /api/reals/ and trigger a file download (CSV or ZIP)
 * in the browser using the response's Content-Disposition filename.
 */
export function useDownloadReals(): UseDownloadRealsResult {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const downloadReals = useCallback(async (params: RealsParams) => {
    setIsLoading(true)
    setError(null)

    try {
      let response

      response = await fetch(`api/weather-generator/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
      })

      console.log(response)

      if (!response.ok) {
        // FastAPI HTTPException returns { detail: "..." }
        let message = `Request failed with status ${response.status}`
        try {
          const errorBody = await response.json()
          message = errorBody.detail ?? message
        } catch {
          // response wasn't JSON, keep default message
        }
        throw new Error(message)
      }

      const blob = await response.blob()

      // Try to pull the filename FastAPI set in Content-Disposition,
      // fall back to a generic name if the header is missing/unparseable.
      const disposition = response.headers.get("Content-Disposition")
      const filenameMatch = disposition?.match(/filename="?([^"]+)"?/)
      const filename = filenameMatch?.[1] ?? "realisations.zip"

      const url = window.URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error occurred")
    } finally {
      setIsLoading(false)
    }
  }, [])

  return { downloadReals, isLoading, error }
}
