import './App.css'
import Card from './Card'
import ProgressCircle from './ProgressCircle'
import { Routes, Route, Link } from 'react-router-dom'
import LessonPage from './LessonPage'

// Each card carries a hardcoded problem sent to the backend as a worksheet image
const cards = [
  { title: 'Long Division', description: 'Practice dividing larger numbers step by step.', variant: 'division' as const, dividend: 247, divisor: 6 },
  { title: 'Times Table', description: 'Build multiplication speed and accuracy.', variant: 'times' as const, dividend: 144, divisor: 12 },
  { title: 'Addition', description: 'Strengthen addition facts and multi-digit sums.', variant: 'addition' as const, dividend: 96, divisor: 8 },
];

// training progress percentage
const trainingProgress = 60; // adjust as needed

function Home() {
  return (
    <div className="homepage">
      <header>
        <h1>Hi InstaLily!</h1>
        <p>Welcome to your homepage.</p>
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
              state={{ dividend: card.dividend, divisor: card.divisor }}
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
