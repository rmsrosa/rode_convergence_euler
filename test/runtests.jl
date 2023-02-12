using Random
using RODEConvergence
using Test

@test RODEConvergence.fG_daviesharte(Xoshiro(123), 1.0, 2^10, 0.2) ≈ RODEConvergence.fG_daviesharte_naive(Xoshiro(123), 1.0, 2^10, 0.2)