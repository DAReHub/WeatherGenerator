import { useState } from "react"

const USERNAME = "demo"
const PASSWORD = "password123"

export default function PasswordGate({
  children,
}: {
  children: React.ReactNode
}) {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [authenticated, setAuthenticated] = useState(false)
  const [error, setError] = useState("")

  function submit(e: React.FormEvent) {
    e.preventDefault()

    if (username === USERNAME && password === PASSWORD) {
      setAuthenticated(true)
    } else {
      setError("Invalid credentials")
    }
  }

  if (authenticated) {
    return children
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
      }}
    >
      <form onSubmit={submit}>
        <h2>Login</h2>

        <input
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <br />

        <input
          placeholder="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <br />

        <button type="submit">Enter</button>

        {error && <p>{error}</p>}
      </form>
    </div>
  )
}
