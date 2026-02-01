# Prover

Inputs to the proving phase are `Vec<Instance>` and `Vec<ConcreteCircuit>`, where `Instance` contains public inputs and `ConcreteCircuit` contains private inputs. The proof is written into `T: TranscriptWrite`.

```rust
pub fn create_proof<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
    ConcreteCircuit: Circuit<C::Scalar>,
>(
    params: &Params<C>,                           // Includes curve points needed for polynomial commitments
    pk: &ProvingKey<C>,                           // Includes fixed column values and permutation info
    circuits: &[ConcreteCircuit],
    instances: &[&[&[C::Scalar]]],
    mut rng: R,
    transcript: &mut T,
) -> Result<(), Error>

```

The following explanation uses the pre–Fiat-Shamir perspective and often mixes columns, column polynomials, and their various representations.

The frequently used `EvaluationDomain` type stores data required for polynomial operations, such as roots of unity and their orders for NTT, the divisors used by NTT, and the Lagrange-basis representation of the vanishing polynomial \\(X^n - 1\\).

In Halo2, "degree" can mean different things:
- The degree of a polynomial in the usual sense.
- The degree of the constraint system, which treats column polynomials (fixed, advice, or instance columns) as degree-1, and refers to the degree of constraint polynomials.
Therefore, the degree of a constraint polynomial is the constraint-system degree multiplied by the number of circuit rows \\(n\\).

Clearly, \\(n\\) Lagrange basis elements are insufficient to represent constraint polynomials, so most polynomials are computed in an extended Lagrange basis using more points.

The term “vanishing polynomial” in Halo2 can be ambiguous. In the following, we use the name consistently but assign different symbols as needed.

## Handling `instances`

In Halo2 terminology, `instance` represents public inputs.

```rust
    let instance: Vec<InstanceSingle<C>> = instances
        .iter()
        .map(|instance| -> Result<InstanceSingle<C>, Error> {
            let instance_values = ...;
            let instance_commitments_projective: Vec<_> = instance_values
                .iter()
                // MSM is used here
                .map(|poly| params.commit_lagrange(poly, Blind::default()))  
                .collect();
            
            ...

            // Send MSM results
            for commitment in &instance_commitments {
                transcript.common_point(*commitment)?;
            }

            let instance_polys: Vec<_> = ...;
            let instance_cosets: Vec<_> = ...;

            Ok(InstanceSingle {
                instance_values,      // Lagrange-basis representation
                instance_polys,       // Coefficient representation
                instance_cosets,      // Extended Lagrange-basis representation
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
```

In this step, `instances` are represented as a polynomial \\(A(X)\\), then:
- \\(A(X)\\) is committed and sent to the verifier
- \\(A(X)\\) is transformed by NTT into coefficient form for later use
- The coefficient form is transformed again to obtain an extended Lagrange-basis representation for later use

## Handling `advices`

In Halo2 terminology, `advice` represents private inputs and internal circuit state.

```rust
    let advice: Vec<AdviceSingle<C>> = circuits
        .iter()
        .zip(instances.iter())
        .map(|(circuit, instances)| -> Result<AdviceSingle<C>, Error> {
            struct WitnessCollection<'a, F: Field> {
                k: u32,
                pub advice: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
                instances: &'a [&'a [F]],
                usable_rows: RangeTo<usize>,
                _marker: std::marker::PhantomData<F>,
            }

            impl<'a, F: Field> Assignment<F> for WitnessCollection<'a, F> {
                ...
            }

            // The last few rows are reserved for blinding factors, so not all rows are usable
            let unusable_rows_start = params.n as usize - (meta.blinding_factors() + 1);

            let mut witness = /* empty `WitnessCollection` */;

            // Run `synthesize` to obtain the circuit’s hidden state, including any private inputs
            ConcreteCircuit::FloorPlanner::synthesize(
                &mut witness,
                circuit,
                config.clone(),
                meta.constants.clone(),
            )?;

            let mut advice = batch_invert_assigned(witness.advice);

            // Add blinding factors to advice columns
            ...

            // Compute commitments to advice column polynomials
            let advice_blinds: Vec<_> = /* random scalars */;
            let advice_commitments_projective: Vec<_> = advice
                .iter()
                .zip(advice_blinds.iter())
                // MSM is used here
                .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
                .collect();
            ...

            // Send MSM results
            for commitment in &advice_commitments {
                transcript.write_point(*commitment)?;
            }

            let advice_polys: Vec<_> = ...;

            let advice_cosets: Vec<_> = ...;

            Ok(AdviceSingle {
                advice_values: advice,  // Lagrange-basis representation
                advice_polys,           // Coefficient representation
                advice_cosets,          // Extended Lagrange-basis representation
                advice_blinds,          // Blinding factors for commitments
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
```

The `advice` flow is similar to `instance`, except that advice values are obtained by executing the circuit.

## Merge Columns Involved in PLOOKUPS Constraints

Halo2 provides lookup constraints based on the PLOOKUPS protocol.

Given expressions for the input columns
    \\[a_1(X), a_2(X), ..., a_n(X)\\]
as \\(A(X)\\), and for the table columns
    \\[s_1(X), a_2(X), ..., s_m(X)\\]
as \\(S(X)\\), the lookup constraint guarantees that every row in \\(A(X)\\) appears in \\(S(X)\\).

In a circuit there may be multiple such constraints. Halo2 merges them using a challenge scalar.

First, the verifier sends the challenge \\(theta\\):
```rust
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();
```

Then the prover executes
```rust
    let lookups: Vec<Vec<lookup::prover::Permuted<C, _>>> = instance_values
        .iter()
        .zip(instance_cosets.iter())
        .zip(advice_values.iter())
        .zip(advice_cosets.iter())
        .map(|(((instance_values, instance_cosets), advice_values), advice_cosets)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup.commit_permuted(
                        pk,
                        params,
                        domain,
                        &value_evaluator,
                        &mut coset_evaluator,
                        theta,
                        advice_values,     // Lagrange-basis representation
                        &fixed_values,
                        instance_values,
                        advice_cosets,     // Extended Lagrange-basis representation
                        &fixed_cosets,
                        instance_cosets,
                        &mut rng,
                        transcript,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;
```

This operation merges multiple lookup constraints into one and commits the merged polynomial.

This may involve any column, including instance, advice, and fixed columns, so they are all passed to `lookup.commit_permuted`. The `value_evaluator` and `coset_evaluator` parameters are used for deferred evaluation.

For deferred evaluation, Halo2 represents polynomials as leaf nodes in an AST; all polynomial expressions are computed on this AST. Concrete polynomials are stored in evaluator objects like `value_evaluator`.

`lookup.commit_permuted` works as follows:

```rust
        // Combine expressions using the challenge theta
        let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
            // For each lookup, compute the Lagrange-basis representation of A(X) or S(X)
            // as an AST
            let unpermuted_expressions: Vec<_> = expressions .iter()
                .map(|expression| { expression.evaluate(...) }) .collect();

            // Same as above, but in the extended Lagrange basis
            let unpermuted_cosets: Vec<_> = expressions .iter()
                .map(|expression| { expression.evaluate(...) }) .collect();

            // If A0(X), ..., An(X), compress to theta^n An(X) + ... + theta A1(X) + A0(X)
            let compressed_expression = unpermuted_expressions.iter().fold(
                poly::Ast::ConstantTerm(C::Scalar::ZERO),
                |acc, expression| &(acc * *theta) + expression,
            );

            // Same as above, but in the extended Lagrange basis
            let compressed_coset = unpermuted_cosets.iter().fold(
                poly::Ast::<_, _, ExtendedLagrangeCoeff>::ConstantTerm(C::Scalar::ZERO),
                |acc, eval| acc * poly::Ast::ConstantTerm(*theta) + eval.clone(),
            );

            (
                // Returns only the AST
                compressed_coset,
                // Evaluate the AST to get a concrete polynomial
                value_evaluator.evaluate(&compressed_expression, domain),
            )
        };

        // Get values of input expressions involved in the lookup and compress them
        let (compressed_input_coset, compressed_input_expression) =
            compress_expressions(&self.input_expressions);

        // Get values of table expressions involved in the lookup and compress them
        let (compressed_table_coset, compressed_table_expression) =
            compress_expressions(&self.table_expressions);

        // Permute compressed (InputExpression, TableExpression) pair
        // Required by PLOOKUPS: the Lagrange coefficients are a permutation of the input polynomial's coefficients.
        let (permuted_input_expression, permuted_table_expression) = permute_expression_pair::<C, _>(
            pk,
            params,
            domain,
            &mut rng,
            &compressed_input_expression,
            &compressed_table_expression,
        )?;

        // Closure to construct commitment to vector of values
        let mut commit_values = |values: &Polynomial<C::Scalar, LagrangeCoeff>| {
            // NTT is used here
            let poly = pk.vk.domain.lagrange_to_coeff(values.clone());
            let blind = Blind(C::Scalar::random(&mut rng));
            let commitment = params.commit_lagrange(values, blind).to_affine();
            (poly, blind, commitment)
        };

        // Commit to permuted polynomials
        let (permuted_input_poly, permuted_input_blind, permuted_input_commitment) =
            commit_values(&permuted_input_expression);

        // Commit to permuted table expression
        let (permuted_table_poly, permuted_table_blind, permuted_table_commitment) =
            commit_values(&permuted_table_expression);

        // Hash permuted input commitment
        transcript.write_point(permuted_input_commitment)?;

        // Hash permuted table commitment
        transcript.write_point(permuted_table_commitment)?;

        // NTT is used here
        let permuted_input_coset = coset_evaluator
            .register_poly(pk.vk.domain.coeff_to_extended(permuted_input_poly.clone()));
        let permuted_table_coset = coset_evaluator
            .register_poly(pk.vk.domain.coeff_to_extended(permuted_table_poly.clone()));

        Ok(Permuted {
            compressed_input_expression, // Compressed input polynomial before permutation (A(X)), Lagrange basis
            compressed_input_coset,      // Extended Lagrange form (unevaluated AST)
            permuted_input_expression,   // Permuted input polynomial, Lagrange basis
            permuted_input_poly,         // Coefficient form
            permuted_input_coset,        // Extended Lagrange form
            permuted_input_blind,        // Blinding scalar
            compressed_table_expression, // Same as above for the table polynomial
            compressed_table_coset,
            permuted_table_expression,
            permuted_table_poly,
            permuted_table_coset,
            permuted_table_blind,
        })
```

## Construct Auxiliary Polynomials for PLOOKUPS and Permutations

To prove lookup and permutation constraints, the prover first constructs auxiliary polynomials \\(z_l(X)\\) and \\(z_p(X)\\).

The verifier sends challenges \\(beta\\) and \\(gamma\\) for the auxiliary polynomials:
```rust
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();alar();
```

The following calls construct and commit the auxiliary polynomials.
```rust
    let permutations: Vec<permutation::prover::Committed<C, _>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| {
            pk.vk.cs.permutation.commit(
                params,
                pk,
                &pk.permutation,
                &advice.advice_values,
                &pk.fixed_values,
                &instance.instance_values,
                beta,
                gamma,
                &mut coset_evaluator,
                &mut rng,
                transcript,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let lookups: Vec<Vec<lookup::prover::Committed<C, _>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            // Construct and commit to products for each lookup
            lookups
                .into_iter()
                .map(|lookup| {
                    lookup.commit_product(
                        pk,
                        params,
                        beta,
                        gamma,
                        &mut coset_evaluator,
                        &mut rng,
                        transcript,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
```

Below is the implementation of `permutation.commit` and `lookup.commit_product`.

The permutation auxiliary polynomial \\(z_p(X)\\) is constructed in the Lagrange basis, then converted to coefficient and extended Lagrange forms.

```rust
        let domain = &pk.vk.domain;

        // How many columns can be included in a single permutation polynomial?
        // We need to multiply by z(X) and (1 - (l_last(X) + l_blind(X))). This
        // will never underflow because of the requirement of at least a degree
        // 3 circuit for the permutation argument.
        assert!(pk.vk.cs_degree >= 3);
        let chunk_len = pk.vk.cs_degree - 2;

        ...

        // For each permutation constraint...
        for (columns, permutations) in self
            .columns
            .chunks(chunk_len)
            .zip(pkey.permutations.chunks(chunk_len))
        {
            // Goal is to compute the products of fractions
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where p_j(X) is the jth column in this permutation,
            // and i is the ith row of the column.

            let mut modified_values = vec![C::Scalar::ONE; params.n as usize];
            // Fill modified_values
            ...

            // The modified_values vector is a vector of products of fractions
            // of the form
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where i is the index into modified_values, for the jth column in
            // the permutation

            // Compute the evaluations of the permutation product polynomial
            // over our domain, starting with z[0] = 1
            let mut z = domain.lagrange_from_vec(z);
            // Set blinding factors
            for z in &mut z[params.n as usize - blinding_factors..] {
                *z = C::Scalar::random(&mut rng);
            }
            // Set new last_z
            ...

            let blind = Blind(C::Scalar::random(&mut rng));
            let permutation_product_blind = blind;

            let permutation_product_commitment_projective = params.commit_lagrange(&z, blind);
            // NTT is used here
            let z = domain.lagrange_to_coeff(z);
            let permutation_product_poly = z.clone();

            // NTT is used here
            let permutation_product_coset =
                evaluator.register_poly(domain.coeff_to_extended(z.clone()));

            let permutation_product_commitment =
                permutation_product_commitment_projective.to_affine();

            // Hash the permutation product commitment
            transcript.write_point(permutation_product_commitment)?;

            sets.push(CommittedSet {
                // Different representations of the auxiliary polynomial
                permutation_product_poly,
                permutation_product_coset,
                permutation_product_blind,
            });
        }

        Ok(Committed { sets })
```

The lookup auxiliary polynomial is constructed similarly, also in the Lagrange basis.

```rust
let blinding_factors = pk.vk.cs.blinding_factors();
        // Goal is to compute the products of fractions
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        //            * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        // Denominator: (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where a_j(X) is the jth input expression in this lookup,
        // where a'(X) is the compression of the permuted input expressions,
        // s_j(X) is the jth table expression in this lookup,
        // s'(X) is the compression of the permuted table expressions,
        // and i is the ith row of the expression.

        // Compute z
        ...

        let product_blind = Blind(C::Scalar::random(rng));
        let product_commitment = params.commit_lagrange(&z, product_blind).to_affine();
        let z = pk.vk.domain.lagrange_to_coeff(z);
        let product_coset = evaluator.register_poly(pk.vk.domain.coeff_to_extended(z.clone()));

        // Hash product commitment
        transcript.write_point(product_commitment)?;

        Ok(Committed::<C, _> {
            permuted: self,
            product_poly: z,
            product_coset,
            product_blind,
        })
```

## Generate the Random Vanishing Polynomial \\(h_{rand}(X)\\)

This is just a random polynomial. It is committed and opened, but appears unrelated to other parts.

## Construct Lookup and Permutation Constraint Polynomials

First, the verifier sends the challenge
```rust
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();
```

Then construct constraint polynomials, all in extended Lagrange form. The `permutation.construct` and `p.construct` calls both take AST leaf nodes and return constructed ASTs.

```rust
    // Evaluate the h(X) polynomial's constraint system expressions for the permutation constraints.
    let (permutations, permutation_expressions): (Vec<_>, Vec<_>) = permutations
        .into_iter()
        .zip(advice_cosets.iter())
        .zip(instance_cosets.iter())
        .map(|((permutation, advice), instance)| {
            permutation.construct(
                pk,
                &pk.vk.cs.permutation,
                advice,
                &fixed_cosets,
                instance,
                &permutation_cosets,  // Permutation polynomials (circuit-specific); the jth Lagrange coeff of the ith poly is delta^i' omega^j'
                                      // representing a permutation from (i, j) to (i', j') in the circuit table
                l0,                   // l_0, l_blind, l_last are Lagrange bases
                l_blind,
                l_last,
                beta,
                gamma,
            )
        })
        .unzip();

    let (lookups, lookup_expressions): (Vec<Vec<_>>, Vec<Vec<_>>) = lookups
        .into_iter()
        .map(|lookups| {
            // Evaluate the h(X) polynomial's constraint system expressions for the lookup constraints, if any.
            lookups
                .into_iter()
                .map(|p| p.construct(beta, gamma, l0, l_blind, l_last))
                .unzip()
        })
        .unzip();
```
The constructed permutation constraints are:

```rust
        let constructed = Constructed { /* data from self */ };

        // Concatenate multiple permutation constraints
        let expressions = iter::empty()
            // Enforce only for the first set.
            // l_0(X) * (1 - z_0(X)) = 0
            .chain(...)
            // Enforce only for the last set.
            // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
            .chain(...)
            // Except for the first set, enforce.
            // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
            .chain(...)
            // And for all the sets we enforce:
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
            // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
            // )
            .chain( ...);
        (
            constructed,
            // Still unevaluated; an AST
            expressions
        )
```

The constructed lookup constraints are:

```rust
        let expressions = iter::empty()
            // l_0(X) * (1 - z(X)) = 0
            .chain(...)
            // l_last(X) * (z(X)^2 - z(X)) = 0
            .chain(...)
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta) (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
            // ) = 0
            .chain(...)
            // Check that the first values in the permuted input expression and permuted
            // fixed expression are the same.
            // l_0(X) * (a'(X) - s'(X)) = 0
            .chain(...)
            // Check that each value in the permuted lookup input expression is either
            // equal to the value above it, or the value at the same index in the
            // permuted table expression.
            // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
            .chain(...);

        (
            Constructed { /* self with a different name */ },
            // Also an AST
            expressions,
        )
```

## Construct the Main Constraint Polynomial

Collect all constraint polynomials and construct the vanishing polynomial \\(h(X)\\).

```rust
    // Concatenate custom constraints with permutation and lookup constraints
    let expressions = advice_cosets
        .iter()
        .zip(instance_cosets.iter())
        .zip(permutation_expressions.into_iter())
        .zip(lookup_expressions.into_iter())
        .flat_map(
            |(((advice_cosets, instance_cosets), permutation_expressions), lookup_expressions)| {
                let fixed_cosets = &fixed_cosets;
                iter::empty()
                    // Custom constraint polynomials
                    .chain(...)
                    // Permutation constraints, if any.
                    .chain(permutation_expressions.into_iter())
                    // Lookup constraints, if any.
                    .chain(lookup_expressions.into_iter().flatten())
            },
        );
    
    // Construct the vanishing polynomial h(X)
    let vanishing = vanishing.construct(
        params,
        domain,
        coset_evaluator,
        expressions,
        y,
        &mut rng,
        transcript,
    )?;
```

\\(h(X)\\) is constructed as follows:

First, use challenge \\(y\\) to combine constraint polynomials \\(P_i(X)\\):
\\[P(X) = \sum y^i P_i(X)\\]

Because P(X) vanishes at all \\(n\\)-th roots of unity \\(\\omega\\),
\\[h(X) = \frac{P(X)}{X^n - 1}\\]

which is a polynomial.

Finally, commit \\(h(X)\\) in pieces.

```rust
        // Evaluate P(X) (in the extended Lagrange basis)
        let h_poly = poly::Ast::distribute_powers(expressions, *y);
        let h_poly = evaluator.evaluate(&h_poly, domain);

        // Compute h(X)
        let h_poly = domain.divide_by_vanishing_poly(h_poly);

        // NTT to convert to coefficient form
        let h_poly = domain.extended_to_coeff(h_poly);

        // Commit in pieces
        ...
        for c in h_commitments.iter() {
            transcript.write_point(*c)?;
        }

        Ok(Constructed {
            h_pieces,
            h_blinds,
            committed: self,
        })
```

## Evaluate and Open Polynomials

The verifier sends the challenge \\(x\\):
```rust
    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
```

The prover evaluates all polynomials at \\(x\\), including instance, advice, fixed, auxiliary polynomials, and \\(h(X)\\), then sends the evaluations to the verifier. Halo2 then uses the multiopen protocol to prove the evaluations are correct.

Multiopen converts openings at multiple points into an opening of a single polynomial at a single point, and then the polynomial commitment scheme proves the evaluations are correct.

- Multiopen protocol: [Halo2 documentation](https://zcash.github.io/halo2/design/proving-system/multipoint-opening.html)
- The polynomial commitment scheme is a variant of the Bulletproofs inner-product protocol:
    - Bulletproofs: *Short Proofs for Confidential Transactions and More*
    - Halo2 variant: *Recursive Proof Composition without a Trusted Setup*
