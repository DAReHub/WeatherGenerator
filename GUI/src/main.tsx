import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import "./index.css"
import App from "./App.tsx"
import { ThemeProvider } from "@/components/theme-provider.tsx"
import PasswordGate from "@/components/PasswordGate"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <PasswordGate>
        <App />
      </PasswordGate>
    </ThemeProvider>
  </StrictMode>
)
