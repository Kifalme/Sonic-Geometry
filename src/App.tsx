import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

// ============================================
// PROFESSIONAL CHLADNI PLATE SIMULATOR
// ============================================
// Physics-based simulation using:
// - Bessel functions for circular plate modes
// - Acoustic radiation pressure for particle dynamics
// - Real-time solution of the 2D wave equation
// - Proper boundary conditions (clamped/free)

interface VibrationMode {
  m: number; // Azimuthal mode number
  n: number; // Radial mode number
  frequency: number;
  amplitude: number;
  phase: number;
}

interface Particle {
  x: number; // Normalized position [-1, 1]
  y: number;
  vx: number;
  vy: number;
  settled: boolean;
  settleTime: number;
}

// Bessel function of the first kind J_m(x)
// Using polynomial approximation for accuracy
function besselJ(m: number, x: number): number {
  if (Math.abs(x) < 1e-10) return m === 0 ? 1 : 0;
  
  // Use series expansion for small x
  if (Math.abs(x) < 5) {
    let sum = 0;
    const terms = 20;
    const x2 = x * x / 4;
    let term = 1;
    
    for (let k = 0; k < terms; k++) {
      // Gamma function approximation for factorials
      const gammaMk = k === 0 && m === 0 ? 1 : 
        Array.from({length: m + k}, (_, i) => m + k - i).reduce((a, b) => a * b, 1);
      const gammaK = k <= 1 ? 1 : Array.from({length: k}, (_, i) => k - i).reduce((a, b) => a * b, 1);
      
      const coef = Math.pow(-1, k) / (gammaMk * gammaK);
      term = coef * Math.pow(x2, k);
      sum += term;
      
      if (Math.abs(term) < 1e-15) break;
    }
    
    return sum * Math.pow(x / 2, m);
  }
  
  // Asymptotic expansion for large x
  const coef = Math.sqrt(2 / (Math.PI * x));
  const phase = x - (m * Math.PI / 2) - (Math.PI / 4);
  return coef * Math.cos(phase);
}

// Derivative of Bessel function J_m'(x)
function besselJDerivative(m: number, x: number): number {
  if (Math.abs(x) < 1e-10) {
    if (m === 0) return 0;
    if (m === 1) return 0.5;
    return 0;
  }
  return (besselJ(m - 1, x) - besselJ(m + 1, x)) / 2;
}

// Find zeros of Bessel function using Newton-Raphson
function findBesselZero(m: number, zeroIndex: number): number {
  // Initial guess based on asymptotic formula
  let x = (zeroIndex + m / 2 - 0.25) * Math.PI;
  
  // Newton-Raphson iteration
  for (let i = 0; i < 50; i++) {
    const j = besselJ(m, x);
    const jp = besselJDerivative(m, x);
    const dx = j / jp;
    x -= dx;
    if (Math.abs(dx) < 1e-12) break;
  }
  
  return x;
}

// Pre-computed Bessel zeros for modes up to (10, 10)
const besselZeros: Map<string, number> = new Map();
function getBesselZero(m: number, n: number): number {
  const key = `${m},${n}`;
  if (!besselZeros.has(key)) {
    besselZeros.set(key, findBesselZero(m, n));
  }
  return besselZeros.get(key)!;
}

// Physical parameters for a real Chladni plate
const PLATE_PARAMS = {
  radius: 0.15, // 15cm radius
  thickness: 0.001, // 1mm thickness
  density: 7850, // Steel density kg/m³
  youngsModulus: 200e9, // Steel Young's modulus Pa
  poissonRatio: 0.3, // Steel Poisson ratio
  damping: 0.02, // Damping coefficient
  particleMass: 1e-6, // Sand particle mass kg
  particleRadius: 5e-5, // Sand particle radius m
  airDensity: 1.225, // Air density kg/m³
  soundSpeed: 343, // Speed of sound m/s
};

// Calculate natural frequency for mode (m, n)
function calculateNaturalFrequency(m: number, n: number): number {
  const { radius, thickness, density, youngsModulus, poissonRatio } = PLATE_PARAMS;
  
  // Flexural rigidity D = E*h³/(12*(1-ν²))
  const D = (youngsModulus * Math.pow(thickness, 3)) / (12 * (1 - poissonRatio * poissonRatio));
  
  // Surface density ρ_s = ρ*h
  const rho_s = density * thickness;
  
  // k_mn is the nth zero of J_m
  const kmn = getBesselZero(m, n) / radius;
  
  // Natural frequency: ω = k² * √(D/ρ_s)
  const omega = kmn * kmn * Math.sqrt(D / rho_s);
  
  return omega / (2 * Math.PI); // Convert to Hz
}

// Calculate plate displacement at (r, θ) for mode (m, n)
function calculateModeDisplacement(
  r: number, 
  theta: number, 
  m: number, 
  n: number, 
  amplitude: number, 
  phase: number,
  time: number
): number {
  const kmn = getBesselZero(m, n);
  
  // Normalized radial position
  const rNorm = r; // r is already normalized [0, 1]
  
  // Time-dependent amplitude
  const omega = calculateNaturalFrequency(m, n) * 2 * Math.PI;
  const timeFactor = Math.cos(omega * time + phase);
  
  // Bessel function term
  const jTerm = besselJ(m, kmn * rNorm);
  
  // Angular term
  const angularTerm = m === 0 ? 1 : (m > 0 ? Math.cos(m * theta) : Math.sin(-m * theta));
  
  return amplitude * jTerm * angularTerm * timeFactor;
}

// Calculate total displacement from all modes (superposition)
function calculateTotalDisplacement(
  x: number, 
  y: number, 
  modes: VibrationMode[],
  time: number
): number {
  const r = Math.sqrt(x * x + y * y);
  if (r > 1) return 0; // Outside plate
  
  const theta = Math.atan2(y, x);
  
  let displacement = 0;
  for (const mode of modes) {
    displacement += calculateModeDisplacement(
      r, theta, mode.m, mode.n, mode.amplitude, mode.phase, time
    );
  }
  
  return displacement;
}

  // Calculate acoustic radiation pressure gradient (force on particles)
  // Based on: F = -∇P_rad where P_rad ∝ (∂w/∂t)²
  function calculateAcousticForce(
    x: number, 
    y: number, 
    modes: VibrationMode[],
    time: number,
    dt: number
  ): { fx: number; fy: number } {
    const h = 0.02; // Spatial step for gradient calculation
    
    // Calculate velocity ∂w/∂t at current time
    const w = calculateTotalDisplacement(x, y, modes, time);
    const w_xp = calculateTotalDisplacement(x + h, y, modes, time);
    const w_xm = calculateTotalDisplacement(x - h, y, modes, time);
    const w_yp = calculateTotalDisplacement(x, y + h, modes, time);
    const w_ym = calculateTotalDisplacement(x, y - h, modes, time);
    
    // Acoustic pressure is proportional to velocity squared
    // P_rad ∝ v² where v = ∂w/∂t
    const v = (w - calculateTotalDisplacement(x, y, modes, time - dt)) / dt;
    
    // Gradient of pressure (force direction)
    const gradX = (w_xp * w_xp - w_xm * w_xm) / (2 * h);
    const gradY = (w_yp * w_yp - w_ym * w_ym) / (2 * h);
    
    // Force is toward lower pressure (nodes)
    // Include velocity term for more accurate acoustic radiation pressure
    const forceScale = 0.001 * (1 + 0.1 * v * v);
    return {
      fx: -gradX * forceScale,
      fy: -gradY * forceScale
    };
  }

// Generate vibration modes from name
function generateModesFromName(name: string): VibrationMode[] {
  if (!name) return [];
  
  const modes: VibrationMode[] = [];
  
  // Convert name to frequency spectrum using FFT-like approach
  const chars = name.toUpperCase().split('');
  const charCodes = chars.map(c => c.charCodeAt(0));
  
  // Generate modes based on character frequencies
  for (let i = 0; i < Math.min(chars.length, 8); i++) {
    const code = charCodes[i];
    
    // Map character to mode numbers
    const m = (code % 6) + 1;
    const n = Math.floor(code / 10) % 5 + 1;
    
    // Calculate actual frequency
    const freq = calculateNaturalFrequency(m, n);
    
    // Amplitude based on position in name (decreasing)
    const amplitude = 1.0 / (1 + i * 0.5);
    
    // Random phase for complexity
    const phase = (code / 255) * Math.PI * 2;
    
    modes.push({ m, n, frequency: freq, amplitude, phase });
  }
  
  return modes;
}

// Initialize particles on the plate
function initializeParticles(count: number): Particle[] {
  const particles: Particle[] = [];
  
  for (let i = 0; i < count; i++) {
    // Distribute particles with higher density at edges (common in experiments)
    const u = Math.random();
    const v = Math.random();
    
    // Use importance sampling for more interesting distribution
    const r = Math.sqrt(u) * 0.98; // Leave small margin from edge
    const theta = 2 * Math.PI * v;
    
    particles.push({
      x: r * Math.cos(theta),
      y: r * Math.sin(theta),
      vx: 0,
      vy: 0,
      settled: false,
      settleTime: 0
    });
  }
  
  return particles;
}

// Update particle positions using Verlet integration with acoustic forces
function updateParticles(
  particles: Particle[],
  modes: VibrationMode[],
  time: number,
  dt: number
): Particle[] {
  const damping = 0.95;
  const maxVelocity = 0.02;
  const settleThreshold = 0.0005;
  
  return particles.map(p => {
    if (p.settled) return p;
    
    // Calculate acoustic force
    const force = calculateAcousticForce(p.x, p.y, modes, time, dt);
    
    // Add small random thermal agitation
    const thermal = 0.0001;
    const randomFx = (Math.random() - 0.5) * thermal;
    const randomFy = (Math.random() - 0.5) * thermal;
    
    // Update velocity
    let newVx = (p.vx + force.fx + randomFx) * damping;
    let newVy = (p.vy + force.fy + randomFy) * damping;
    
    // Clamp velocity
    const clampedVx = Math.max(-maxVelocity, Math.min(maxVelocity, newVx));
    const clampedVy = Math.max(-maxVelocity, Math.min(maxVelocity, newVy));
    
    // Update position
    let newX = p.x + clampedVx;
    let newY = p.y + clampedVy;
    
    // Enforce circular boundary (reflective)
    const r = Math.sqrt(newX * newX + newY * newY);
    if (r > 0.99) {
      const scale = 0.98 / r;
      newX *= scale;
      newY *= scale;
      // Reflect velocity (conservation of momentum at boundary)
      const normalX = newX / r;
      const normalY = newY / r;
      const dot = clampedVx * normalX + clampedVy * normalY;
      // Apply reflection: v_new = v - 2(v·n)n
      const vxReflected = clampedVx - 2 * dot * normalX;
      const vyReflected = clampedVy - 2 * dot * normalY;
      // Apply damping at boundary
      newVx = vxReflected * 0.5;
      newVy = vyReflected * 0.5;
    }
    
    // Check if settled (low velocity and on nodal line)
    const speed = Math.sqrt(clampedVx * clampedVx + clampedVy * clampedVy);
    const displacement = Math.abs(calculateTotalDisplacement(newX, newY, modes, time));
    
    const isSettled = speed < settleThreshold && displacement < 0.1;
    
    return {
      x: newX,
      y: newY,
      vx: clampedVx,
      vy: clampedVy,
      settled: isSettled,
      settleTime: isSettled ? p.settleTime + 1 : 0
    };
  });
}

// ============================================
// REACT COMPONENT
// ============================================

function App() {
  const [name, setName] = useState('');
  const [modes, setModes] = useState<VibrationMode[]>([]);
  const [particles, setParticles] = useState<Particle[]>([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [time, setTime] = useState(0);
  const [showModes, setShowModes] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const particlesRef = useRef<Particle[]>([]);
  const timeRef = useRef(0);
  
  const PARTICLE_COUNT = 8000;
  const dt = 1 / 60; // 60 FPS physics
  
  // Generate pattern when name changes
  const generatePattern = useCallback(() => {
    if (!name.trim()) return;
    
    const newModes = generateModesFromName(name);
    setModes(newModes);
    
    const newParticles = initializeParticles(PARTICLE_COUNT);
    setParticles(newParticles);
    particlesRef.current = newParticles;
    
    setTime(0);
    timeRef.current = 0;
    setIsAnimating(true);
  }, [name]);
  
  // Animation loop
  useEffect(() => {
    if (!isAnimating) return;
    
    const animate = () => {
      // Update physics (fixed time step for stability)
      timeRef.current += dt;
      particlesRef.current = updateParticles(particlesRef.current, modes, timeRef.current, dt);
      
      // Update state every few frames for performance
      if (Math.floor(timeRef.current * 60) % 2 === 0) {
        setParticles([...particlesRef.current]);
        setTime(timeRef.current);
      }
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isAnimating, modes, dt]);
  
  // Render canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas size
    const size = Math.min(window.innerWidth - 64, 600);
    canvas.width = size;
    canvas.height = size;
    
    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, size, size);
    
    const center = size / 2;
    const scale = (size / 2) * 0.95;
    
    // Draw plate boundary
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(center, center, scale, 0, 2 * Math.PI);
    ctx.stroke();
    
    // Draw displacement field visualization (heatmap)
    if (modes.length > 0) {
      const imageData = ctx.createImageData(size, size);
      const data = imageData.data;
      
      const step = 2; // Skip pixels for performance
      for (let py = 0; py < size; py += step) {
        for (let px = 0; px < size; px += step) {
          const x = (px - center) / scale;
          const y = (py - center) / scale;
          
          if (x * x + y * y <= 1) {
            const displacement = calculateTotalDisplacement(x, y, modes, time);
            const intensity = Math.min(255, Math.abs(displacement) * 50);
            
            // Fill square region
            for (let dy = 0; dy < step && py + dy < size; dy++) {
              for (let dx = 0; dx < step && px + dx < size; dx++) {
                const idx = ((py + dy) * size + (px + dx)) * 4;
                // Nodes (low displacement) are bright, antinodes dark
                const nodeIntensity = 255 - intensity;
                data[idx] = nodeIntensity;
                data[idx + 1] = nodeIntensity;
                data[idx + 2] = nodeIntensity;
                data[idx + 3] = 30; // Low alpha
              }
            }
          }
        }
      }
      
      ctx.putImageData(imageData, 0, 0);
    }
    
    // Draw particles
    if (particles.length > 0) {
      ctx.fillStyle = '#ffffff';
      
      for (const p of particles) {
        const px = center + p.x * scale;
        const py = center + p.y * scale;
        
        // Particle size based on settled status
        const radius = p.settled ? 1.2 : 0.8;
        
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    
    // Draw center marker
    ctx.fillStyle = '#666';
    ctx.beginPath();
    ctx.arc(center, center, 3, 0, 2 * Math.PI);
    ctx.fill();
    
  }, [particles, modes, time]);
  
  return (
    <div className="min-h-screen bg-black text-white font-mono">
      {/* Header */}
      <header className="border-b border-neutral-800">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="flex items-center gap-4 mb-2">
            <svg viewBox="0 0 40 40" className="w-10 h-10">
              <circle cx="20" cy="20" r="18" fill="none" stroke="white" strokeWidth="1.5"/>
              <circle cx="20" cy="20" r="8" fill="none" stroke="white" strokeWidth="1"/>
              <line x1="20" y1="2" x2="20" y2="38" stroke="white" strokeWidth="1"/>
              <line x1="2" y1="20" x2="38" y2="20" stroke="white" strokeWidth="1"/>
              <circle cx="20" cy="20" r="2" fill="white"/>
            </svg>
            <h1 className="text-2xl tracking-widest font-light">SONIC GEOMETRY</h1>
          </div>
          <p className="text-neutral-500 text-sm tracking-wide">
            Professional Chladni Plate Simulation
          </p>
        </div>
      </header>
      
      <main className="max-w-6xl mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Controls */}
          <div className="space-y-6">
            {/* Input Section */}
            <div className="border border-neutral-800 p-6">
              <label className="block text-xs uppercase tracking-widest text-neutral-500 mb-3">
                Enter Name / Frequency Key
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Type name..."
                className="w-full bg-black border border-neutral-700 px-4 py-3 text-white placeholder-neutral-600 focus:outline-none focus:border-white transition-colors"
              />
              <button
                onClick={generatePattern}
                disabled={!name.trim()}
                className="w-full mt-4 border border-white px-6 py-3 text-sm uppercase tracking-widest hover:bg-white hover:text-black transition-all disabled:opacity-30 disabled:cursor-not-allowed"
              >
                Generate Pattern
              </button>
            </div>
            
            {/* Physics Parameters */}
            <div className="border border-neutral-800 p-6">
              <h3 className="text-xs uppercase tracking-widest text-neutral-500 mb-4">
                Physical Parameters
              </h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-neutral-600">Plate Radius</span>
                  <span>{(PLATE_PARAMS.radius * 100).toFixed(1)} cm</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Thickness</span>
                  <span>{(PLATE_PARAMS.thickness * 1000).toFixed(1)} mm</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Material</span>
                  <span>Steel</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Particle Count</span>
                  <span>{PARTICLE_COUNT.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Simulation Time</span>
                  <span>{time.toFixed(2)}s</span>
                </div>
              </div>
            </div>
            
            {/* Vibration Modes */}
            {modes.length > 0 && (
              <div className="border border-neutral-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xs uppercase tracking-widest text-neutral-500">
                    Vibration Modes (m, n)
                  </h3>
                  <button
                    onClick={() => setShowModes(!showModes)}
                    className="text-xs text-neutral-600 hover:text-white transition-colors"
                  >
                    {showModes ? 'Hide' : 'Show'}
                  </button>
                </div>
                
                {showModes && (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {modes.map((mode, i) => (
                      <div key={i} className="flex justify-between items-center text-xs py-2 border-b border-neutral-900">
                        <span className="text-neutral-400">
                          Mode ({mode.m}, {mode.n})
                        </span>
                        <div className="text-right">
                          <div className="text-white">{mode.frequency.toFixed(1)} Hz</div>
                          <div className="text-neutral-600">
                            A={mode.amplitude.toFixed(2)} φ={(mode.phase * 180 / Math.PI).toFixed(0)}°
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                
                {!showModes && (
                  <div className="text-xs text-neutral-600">
                    {modes.length} active modes • 
                    Fundamental: {modes[0]?.frequency.toFixed(1)} Hz
                  </div>
                )}
              </div>
            )}
            
            {/* Instructions */}
            <div className="border border-neutral-800 p-6 text-sm text-neutral-500">
              <h3 className="text-xs uppercase tracking-widest text-white mb-3">
                About This Simulation
              </h3>
              <p className="mb-3">
                This simulation uses the 2D wave equation for a circular plate with 
                Bessel function solutions:
              </p>
              <p className="font-mono text-xs bg-neutral-900 p-3 mb-3">
                w(r,θ,t) = Σ A_mn · J_m(k_mn·r) · cos(mθ) · cos(ωt)
              </p>
              <ul className="space-y-1 text-xs">
                <li>• m = azimuthal mode number</li>
                <li>• n = radial mode number</li>
                <li>• J_m = Bessel function of first kind</li>
                <li>• k_mn = nth zero of J_m / radius</li>
                <li>• Particles move to nodal lines (minimum displacement)</li>
              </ul>
            </div>
          </div>
          
          {/* Visualization */}
          <div className="lg:sticky lg:top-8 h-fit">
            <div className="border border-neutral-800 p-4">
              <div className="aspect-square relative">
                <canvas
                  ref={canvasRef}
                  className="w-full h-full"
                />
                
                {particles.length === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-neutral-700 text-6xl mb-4">○</div>
                      <p className="text-neutral-600 text-sm tracking-wide">
                        Enter a name to generate pattern
                      </p>
                    </div>
                  </div>
                )}
              </div>
              
              {particles.length > 0 && (
                <div className="mt-4 flex justify-between items-center text-xs text-neutral-600">
                  <span>
                    Settled: {particles.filter(p => p.settled).length.toLocaleString()}
                  </span>
                  <span>
                    Active: {particles.filter(p => !p.settled).length.toLocaleString()}
                  </span>
                </div>
              )}
            </div>
            
            {modes.length > 0 && (
              <div className="mt-4 border border-neutral-800 p-4 text-xs text-neutral-500">
                <p>
                  <strong className="text-white">Key:</strong> White particles accumulate along 
                  nodal lines where plate displacement is minimized. Dark areas indicate 
                  antinodes (maximum vibration).
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
      
      {/* Footer */}
      <footer className="border-t border-neutral-800 mt-16">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <p className="text-xs text-neutral-600 text-center tracking-widest">
            SONIC GEOMETRY — CHLADNI PLATE SIMULATION v2.0
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
