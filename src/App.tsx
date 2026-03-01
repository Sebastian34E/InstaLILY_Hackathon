import './App.css'
import Card from './Card'
import ProgressCircle from './ProgressCircle'
import { Routes, Route, Link } from 'react-router-dom'
import LessonPage from './LessonPage'

// Single worksheet card — problems loaded from backend PDF at runtime
const cards = [
  { title: 'Long Division', description: 'Practice dividing larger numbers step by step.', variant: 'division' as const },
];

// training progress percentage
const trainingProgress = 60; // adjust as needed

function Home() {
  return (
    <div className="homepage">
      <header>
        <h1>Welcome to InstaTutor!</h1>
      </header>
      <section className="progress-row">
        <ProgressCircle label="Training Progress" percent={trainingProgress} />
      </section>
      <section className="card-section">
        <h2>Worksheet</h2>
        <div className="card-grid">
          {cards.map((card) => (
            <Link
              key={card.title}
              to="/lesson"
              style={{ textDecoration: 'none' }}
            >
              <Card
                title={card.title}
                description={card.description}
                variant={card.variant}
              />
            </Link>
          ))}
        </div>
      </section>
    </div>
  )
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/lesson" element={<LessonPage />} />
    </Routes>
  )
}

export default App
