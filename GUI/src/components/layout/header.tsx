import { Button } from "@/components/ui/button"
import { ModeToggle } from "@/components/mode-toggle"
import { CloudLightning } from "lucide-react"

interface HeaderProps {
  onAboutClick?: () => void
}

export function Header({ onAboutClick }: HeaderProps) {
  return (
    <header className="rounded-2xl border bg-sidebar shadow-lg backdrop-blur-sm ">
      <div className="flex h-16 w-full items-center justify-between px-4">
        {/* Logo / Title */}
        <div className="flex items-center gap-2">
          <>
            <svg width="0" height="0" className="absolute">
              <defs>
                <linearGradient
                  id="cloudGradient"
                  x1="0%"
                  y1="0%"
                  x2="0%"
                  y2="100%"
                >
                  <stop offset="0%" stopColor="#38bdf8" /> {/* Sky blue */}
                  <stop offset="100%" stopColor="#ef4444" /> {/* Red */}
                </linearGradient>
              </defs>
            </svg>

            <CloudLightning
              className="h-10 w-10"
              style={{ stroke: "url(#cloudGradient)" }}
            />
          </>
          <span className="text-2xl font-bold tracking-tight">
            UK Weather Generator
          </span>
        </div>

        {/* Nav - same for desktop and mobile since it's minimal */}
        <div className="flex items-center gap-2">
          <ModeToggle />
          <Button variant="ghost" onClick={onAboutClick}>
            About
          </Button>
        </div>
      </div>
    </header>
  )
}
