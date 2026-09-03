import '@testing-library/jest-dom'

// jsdom has no `ResizeObserver` - needed by any component using recharts'
// `ResponsiveContainer` (Academy's `EpsilonDecayChart`/`PPOClipChart`/
// `DiscountChart`/`TwoHotBinsChart` and friends), which otherwise throws
// at render time in every test that mounts one.
if (typeof globalThis.ResizeObserver === 'undefined') {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  globalThis.ResizeObserver = ResizeObserverStub
}
