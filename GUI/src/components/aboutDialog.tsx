import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog.tsx"
import { Separator } from "@/components/ui/separator"
import { Button } from "@/components/ui/button.tsx"
import { SquareArrowOutUpRight } from "lucide-react"

export function AboutDialog({
  aboutOpen,
  setAboutOpen,
}: {
  aboutOpen: boolean
  setAboutOpen: (open: boolean) => void
}) {
  return (
    <Dialog open={aboutOpen} onOpenChange={setAboutOpen}>
      <DialogContent className="h-[80vh] sm:max-w-5xl">
        <DialogHeader>
          <DialogTitle className="text-2xl">
            Welcome to the DARe Hub and Climate Co-Centre Weather Generator!
          </DialogTitle>
        </DialogHeader>
        <div className="-mx-4 no-scrollbar space-y-4 overflow-y-auto px-4 text-sm leading-relaxed">
          <h2 className="text-xl font-semibold tracking-wide">
            What is a Weather Generator?
          </h2>
          <p>
            Climate models help us understand future climate, but their
            resolution is too coarse for local-scale assessment. Downscaling
            tools such as weather generators (WGs) address this by translating
            climate model projections to finer scales. WGs are statistical
            models trained on observed climate data (e.g., gauge-based records)
            to produce synthetic climate series with similar statistical
            properties to the observed data for a specific location. These
            synthetic series are then perturbed with climate change signals from
            climate models to generate downscaled future climate projections.
          </p>
          <p>
            Within the DARe Hub and the Climate + Co-Centre, a WG has been
            developed that downscales projections from the Met Office's UKCP18
            dataset, calibrated against reference observations. UKCP18 covers 18
            variables (including precipitation, temperature, and windspeed) out
            to 2080, and the WG's downscaled outputs provide multiple plausible
            future scenarios for use in impact assessment models.
          </p>
          <Button asChild>
            <a
              href="https://dare.ac.uk/wp-content/uploads/2026/01/DARe_One-page-summary_Weather-generator_Mar-2026.pdf"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn more! <SquareArrowOutUpRight />
            </a>
          </Button>
          <Button asChild>
            <a
              href="https://dare.ac.uk/"
              target="_blank"
              rel="noopener noreferrer"
            >
              DARe <SquareArrowOutUpRight />
            </a>
          </Button>
          <Button asChild>
            <a
              href="https://climatecocentre.org/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Climate + Co-Centre <SquareArrowOutUpRight />
            </a>
          </Button>
          <Separator />
          <h2 className="text-xl font-semibold tracking-wide">
            How to use this tool?
          </h2>
          <p>
            To generate an ensemble, simply select: a grid cell on the map, a
            UKCP18 ensemble member, climate variable, future time period,
            temporal resolution, and the number of realisations you would like.
            Then press Generate, and wait for processing and download to
            complete.
          </p>
          <Separator />
          <h2 className="text-xl font-semibold tracking-wide">FAQs</h2>
          <h3 className="text-lg">What is a Realisation?</h3>
          <p>
            A realisation is a single time series containing two attributes –
            date/time and a value of interest e.g. rainfall or temperature. A
            set of realisations is known as Ensemble.
          </p>
          <h3 className="text-lg">
            How is the weather generator related to UKCP18?
          </h3>
          <p>
            The weather generator uses 'Change Factors' to relate the
            projections from UKCP18 to observations. We fit our Weather
            Generator (WG) to a reference dataset which is obtained either from
            gauges (maintained by EA, SEPA, NRW/Met office) or gridded datasets
            (such as HadUK). Once confidence is established in the ability of
            the WG to simulate past weather, we then derive projections into the
            future. To derive future projections, we use change factors (or
            climate change signals) from UKCP18 to re-fit the WG by tweaking
            observations of the past. This ensures that future climate
            projections are consistent (not sporadic or unrealistic) with
            observations.
          </p>
          <h3 className="text-lg">
            What does the weather generator provide which UKCP18 does not?
          </h3>
          <p>
            UKCP18 projections typically provide a limited set of models (i.e.
            realisations) which may not adequately capture uncertainties
            associated with future climate projections. We therefore need a
            greater number of realisations which can encompass a large range of
            possible future weather states. The WG can produce a larger set of
            user-defined realisations which are not readily available within
            UKCP18 projections.
          </p>
          <h3 className="text-lg">How can I process my own realisations?</h3>
          <p>
            Check out our{" "}
            <a
              href="https://github.com/DAReHub/WeatherGenerator"
              target="_blank"
              className="text-blue-600 hover:text-blue-700 hover:underline"
            >
              Weather Generator repository!
            </a>
          </p>
          <Separator />
          <h3 className="text-xl font-semibold tracking-wide">
            Roadmap
          </h3>
          <ul className="list-inside list-disc">
            <li>A higher resolution grid at 1 km.</li>
            <li>Increased limits on realisation count.</li>
            <li>Additional climate variables.</li>
            <li>
              Correlated weather output over multiple grid cell selections.
            </li>
          </ul>
          <Separator />
          <h3 className="text-xl font-semibold tracking-wide">Limitations</h3>
          <p>
            Currently, the Weather Generator is configured for single-site
            simulations only and therefore outputs generated from multiple cells
            will be uncorrelated. Any outputs generated from a cell should be
            used in isolation. However, if spatially correlated weather output
            (specifically rainfall) is needed, please reach out to
            azhar.mohammed@newcastle.ac.uk.
          </p>
          <Separator />
          <h3 className="text-xl font-semibold tracking-wide">Disclaimer</h3>
          <p>
            The data generated by this site is synthetic and provided for
            informational and illustrative purposes only. It does not represent
            an actual weather forecast and may not accurately reflect future
            weather conditions. Users may choose to use the generated data for
            any purpose; however, to the fullest extent permitted by law, we
            accept no liability for any loss, damage, injury, or other
            consequence arising from such use or reliance upon the data.
          </p>
          <Separator />
          <div className="flex flex-col items-center justify-center gap-8 pt-2 sm:flex-row sm:gap-16">
            <img
              src="NU_DARe_Master Logo_Digital_Secondary Logo_Green and Blue.png"
              className="h-30 w-auto object-contain"
              alt="DARe logo"
            />
            <img
              src="cocentre_logo_transparent.png"
              className="h-30 w-auto object-contain"
              alt="Co-Centre logo"
            />
          </div>
          <Separator />
        </div>
      </DialogContent>
    </Dialog>
  )
}