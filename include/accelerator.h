#pragma once

#include <invert_quda.h>

namespace quda
{

  /**
    @brief Generic accelerated solver with an accelerator. Let me explain what accelerate means here:
    - Solvers are used in iterations based on an operator and a vector, e.g. to solver A * x = b, we have
        b = Solver[A, x].
    - Now assume we have another set of operator B and vector c and y. The acceleration works as
        y = forward(x) -> c = Solver[B, y] -> b = backward(c),
      as long as Solver[B, y] is faster than Solver[A, x], a speed up might be achieved.
    - As an analogy to multi-grid, forward is the restrictor, backward is the prolongator, and Solver is
      the coarse grid solve.
    - In this class, the overloaded operator() performs the above `acceleration`.
    - The method train_param provides the transformer_t class the original operator A (`matPrecon`), the
      Solver (`base_solver`), and another solver object that can be used to generate null space vector.
      The transformer_t object is then expected to use these to train the parameters to perform better
      forward and backward transformations.
   */
  template <class transformer_t, class solver_t> class AcceleratedSolver : public Solver
  {

    bool active_training = false;

    std::unique_ptr<solver_t> base_solver;
    std::unique_ptr<solver_t> ref_solver; // Here we declare a copy of the solver to avoid temporary buffer collisions.

    const DiracMatrix &matPrecon;

    transformer_t transformer;

  public:
    AcceleratedSolver(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                      const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile) :
      Solver(mat, matSloppy, matPrecon, matEig, param, profile), matPrecon(matPrecon), transformer(param, profile)
    {
      base_solver = std::make_unique<solver_t>(mat, matSloppy, matPrecon, matEig, param, profile);
      ref_solver = std::make_unique<solver_t>(mat, matSloppy, matPrecon, matEig, param, profile);
    }

    /**
     * @brief Forward transform, run base solver (e.g. CG), then backward transform.
     * @param out Solution vector.
     * @param in Right-hand side.
     */
    virtual void operator()(ColorSpinorField &out, ColorSpinorField &in)
    {
      if (transformer.trained) {
        transformer.apply(*base_solver, out, in);
      } else {
        ref_solver->operator()(out, in);
      }
    }

    virtual bool hermitian() { return base_solver->hermitian(); }

    /**
     * @brief Train the underlying accelerate parameter.
     * @param null Solver to solve for null vectors.
     * @param in meta color spinor field.
     */
    virtual void train_param(Solver &null, ColorSpinorField &in)
    {
      if (!active_training && !transformer.trained) {
        active_training = true;
        pushVerbosity(param.verbosity_precondition);
        transformer.train(matPrecon, *base_solver, null, in);
        popVerbosity();
        active_training = false;
      }
    }
  };

} // namespace quda
