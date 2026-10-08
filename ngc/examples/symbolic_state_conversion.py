"""Small SymPy prototype: implicit equations -> explicit state derivatives.

From ngc/: uv run --no-project --with sympy python examples/symbolic_state_conversion.py

Inputs are SymPy expressions equal to zero, with explicitly declared symbols.
This prototype handles square systems affine in the state derivatives. It reports
algebraic conditions; units, frame definitions, and physical assumptions remain
part of the model specification. SymPy expressions may already be simplified:
record the source model's original domain separately when transcribing formulas.
"""
import sympy as sp
from sympy.solvers.solveset import NonlinearError


def solve_derivatives(residuals, derivatives, declared_symbols):
    """Solve M(z,t) z_dot = b(z,t), retaining conditions for this ODE chart.

    A singular system is reported for constraint/DAE analysis. Such a system can
    be a meaningful model even though an explicit derivative is unavailable.
    """
    residuals = sp.Matrix(residuals)
    derivatives = tuple(derivatives)
    if residuals.cols != 1 or not derivatives:
        raise ValueError('Provide a column of residuals and derivative symbols.')
    if len(set(derivatives)) != len(derivatives):
        raise ValueError('Each derivative must have its own symbol.')
    unknown = residuals.free_symbols - set(derivatives) - set(declared_symbols)
    if unknown:
        raise ValueError(f'Undeclared symbols: {sorted(map(str, unknown))}')
    if residuals.rows != len(derivatives):
        return {'status': 'constraint_analysis_required',
                'reason': 'Equation count differs from derivative count.'}
    try:
        mass, forcing = sp.linear_eq_to_matrix(residuals, derivatives)
    except NonlinearError:
        return {'status': 'nonlinear_derivative_system',
                'reason': 'A branch or implicit numerical solve must be specified.'}
    determinant = sp.trigsimp(mass.det())
    if determinant == 0:
        return {'status': 'constraint_analysis_required', 'mass_matrix': mass,
                'reason': 'The derivative coefficient matrix is singular.'}
    rhs = mass.inv().multiply(forcing).applyfunc(sp.simplify).applyfunc(sp.trigsimp)
    substitution = dict(zip(derivatives, rhs))
    residual_check = residuals.subs(substitution, simultaneous=True).applyfunc(sp.simplify).applyfunc(sp.trigsimp)
    # Include denominators as well as the chart's nonzero determinant. Further
    # source assumptions (positive lengths, branch intervals, etc.) are supplied
    # by the author; they cannot all be recovered from simplified expressions.
    conditions = {sp.Ne(determinant, 0)}
    for expression in list(residuals) + list(rhs):
        conditions.add(sp.Ne(expression.as_numer_denom()[1], 0))
    conditions.discard(sp.true)
    return {'status': 'explicit_on_domain', 'mass_matrix': mass,
            'rhs': rhs, 'conditions': sorted(conditions, key=str),
            'residual_check': residual_check,
            'identity_verified': all(value == 0 for value in residual_check)}


def change_coordinates(old_states, old_rhs, new_states, old_from_new, time, parameters=()):
    """For x=h(z,t), solve h_z z_dot = f(h(z,t),t) - h_t.

    old_from_new supplies the transformation explicitly, so no inverse branch
    is guessed by the program. Its Jacobian determines local invertibility.
    """
    old_states, new_states = tuple(old_states), tuple(new_states)
    old_rhs, mapping = sp.Matrix(old_rhs), sp.Matrix(old_from_new)
    if old_rhs.shape != (len(old_states), 1) or mapping.shape != old_rhs.shape:
        raise ValueError('The old state, vector field, and mapping must have equal dimensions.')
    if len(old_states) != len(new_states):
        raise ValueError('This coordinate-change prototype uses equal state dimensions.')
    derivatives = sp.symbols(f'dz0:{len(new_states)}', real=True)
    replaced_rhs = old_rhs.subs(dict(zip(old_states, mapping)), simultaneous=True)
    residuals = mapping.jacobian(new_states) * sp.Matrix(derivatives) + mapping.diff(time) - replaced_rhs
    return solve_derivatives(residuals, derivatives, (*new_states, time, *parameters))


def demo():
    t = sp.Symbol('t', real=True)
    x, y, radius, angle, speed_x, speed_y = sp.symbols('x y r phi a b', real=True)
    # A point translating in a plane. a and b are Cartesian velocity components.
    # The polar chart uses r > 0 and an explicitly chosen local angle interval.
    mapping = [radius * sp.cos(angle), radius * sp.sin(angle)]
    result = change_coordinates([x, y], [speed_x, speed_y], [radius, angle], mapping, t,
                                parameters=[speed_x, speed_y])
    print('Cartesian -> polar:', result)
    # At (x,y)=(3,4), velocity=(2,-1): r_dot=0.4 and phi_dot=-0.44.
    values = {radius: 5, angle: sp.atan2(4, 3), speed_x: 2, speed_y: -1}
    numeric_rhs = result['rhs'].subs(values).evalf()
    print('Example derivative [r_dot, phi_dot]:', numeric_rhs)
    recovered = sp.Matrix(mapping).jacobian([radius, angle]) * result['rhs']
    print('Recovered Cartesian velocity:', recovered.applyfunc(sp.trigsimp))
    # A translating origin illustrates the explicit time derivative h_t.
    moving = change_coordinates([x], [speed_x], [radius], [radius + speed_y*t], t,
                                parameters=[speed_x, speed_y])
    print('Moving origin derivative:', moving['rhs'])

    dx, dy, typo = sp.symbols('dx dy typo')
    cases = {
        'missing equation': [dx - speed_x],
        'dependent equations': [dx + dy, 2*dx + 2*dy],
        'nonlinear derivatives': [dx**2 - 1, dy - speed_y],
        'undeclared symbol': [dx - typo, dy - speed_y],
    }
    for name, equations in cases.items():
        try:
            print(name + ':', solve_derivatives(equations, [dx, dy], [speed_x, speed_y]))
        except ValueError as error:
            print(name + ':', error)


if __name__ == '__main__':
    demo()
