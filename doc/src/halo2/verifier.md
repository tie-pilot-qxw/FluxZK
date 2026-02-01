# Verifier

The verifier can construct the correct main constraint polynomial $P(X)$ from the public circuit information. After receiving the prover’s evaluations at $x$ for all polynomials (instance, advice, fixed, auxiliary polynomials, and $h(X)$), the verifier can compute $P(x)$. Then, for the prover-provided $h(x)$, the verifier checks whether
$ P(x) = (x^n - 1) h(x) $
holds.

If all submitted polynomial evaluations are correct, the verifier accepts the proof. This is the role of the polynomial commitment protocol.
