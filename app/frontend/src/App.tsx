import { Button } from "@/components/ui/button"

// Placeholder shell — T06 replaces this with the real phase router (LiveScreen /
// ReportScreen). It exists so T01 has a smoke-tested page importing a shadcn
// component, and so `npm run dev` renders something.
function App() {
  return (
    <main className="flex min-h-svh flex-col items-center justify-center gap-4">
      <h1 className="text-2xl font-semibold">ORC demo</h1>
      <p className="text-muted-foreground">Scaffold ready.</p>
      <Button>Start</Button>
    </main>
  )
}

export default App
