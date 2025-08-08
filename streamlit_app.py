# streamlit_app.py
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Set

import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

x = sp.symbols('x')
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

RULE_OPTIONS = ["Power", "Product", "Quotient", "Chain"]

@dataclass
class Problem:
    expr: sp.Expr
    rules: Set[str]
    level: int

# -----------------------------
# Problem generation
# -----------------------------

def rand_nonzero_int(a: int, b: int, exclude: Set[int] = {0}) -> int:
    while True:
        n = random.randint(a, b)
        if n not in exclude:
            return n


def gen_power_only() -> Problem:
    # Simple a*x^n, n integer >=1
    a = rand_nonzero_int(-6, 6)
    n = rand_nonzero_int(1, 8)
    expr = a * x**n
    return Problem(expr=sp.simplify(expr), rules={"Power"}, level=1)


def gen_chain_simple() -> Problem:
    # (ax + b)^n with n>=2
    a = rand_nonzero_int(-5, 5)
    b = random.randint(-6, 6)
    n = rand_nonzero_int(2, 6)
    expr = (a*x + b)**n
    return Problem(expr=sp.simplify(expr), rules={"Chain"}, level=2)


def random_quadratic() -> sp.Expr:
    A = rand_nonzero_int(-3, 3)
    B = random.randint(-5, 5)
    C = random.randint(-4, 4)
    return A*x**2 + B*x + C


def gen_product_basic() -> Problem:
    # (x^a + c)*(d*x^b + e)
    a = rand_nonzero_int(1, 4)
    b = rand_nonzero_int(1, 4)
    c = random.randint(-6, 6)
    d = rand_nonzero_int(-5, 5)
    e = random.randint(-6, 6)
    expr = (x**a + c) * (d*x**b + e)
    return Problem(expr=sp.simplify(expr), rules={"Product"}, level=2)


def gen_quotient_basic() -> Problem:
    # (x^a + c) / (d*x^b + e) where denominator nonzero form
    a = rand_nonzero_int(1, 4)
    b = rand_nonzero_int(1, 4)
    c = random.randint(-5, 5)
    d = rand_nonzero_int(-5, 5)
    e = random.randint(-5, 5)
    expr = (x**a + c) / (d*x**b + e)
    return Problem(expr=sp.simplify(expr), rules={"Quotient"}, level=2)


def gen_product_chain() -> Problem:
    # (quadratic)^n * (linear)
    n = rand_nonzero_int(2, 4)
    expr = (random_quadratic())**n * (rand_nonzero_int(1,4)*x + random.randint(-5,5))
    return Problem(expr=sp.simplify(expr), rules={"Product", "Chain"}, level=3)


def gen_quotient_chain() -> Problem:
    # (quadratic)^n / (linear)^m
    n = rand_nonzero_int(2, 4)
    m = rand_nonzero_int(1, 3)
    expr = (random_quadratic())**n / (rand_nonzero_int(1,4)*x + random.randint(-5,5))**m
    return Problem(expr=sp.simplify(expr), rules={"Quotient", "Chain"}, level=3)


def make_problem(difficulty: str) -> Problem:
    # Weighted random by difficulty
    if difficulty == "Beginner":
        choices = [gen_power_only]*3 + [gen_chain_simple]*2 + [gen_product_basic] + [gen_quotient_basic]
    elif difficulty == "Intermediate":
        choices = [gen_power_only, gen_chain_simple, gen_product_basic, gen_quotient_basic, gen_product_chain]
    else:
        choices = [gen_product_chain, gen_quotient_chain, gen_product_basic, gen_quotient_basic, gen_chain_simple]
    return random.choice(choices)()

# -----------------------------
# Hints & explanations
# -----------------------------

def hint_for(problem: Problem) -> str:
    f = problem.expr
    rules = problem.rules
    if rules == {"Power"}:
        a = sp.simplify(sp.together(f/x**sp.degree(f, x))) if sp.degree(f, x) else 1
        n = sp.degree(f, x)
        return f"Power rule: d/dx[a*x^n] = a*n*x^(n-1). Here a={sp.simplify(a)}, n={n}."
    if rules == {"Chain"}:
        u = None
        # crude: try to match (ax+b)^n
        if f.is_Pow and f.args[0].is_Add:
            u = sp.Symbol('u')
            return r"Chain rule: if y = [g(x)]^n then dy/dx = n[g(x)]^{n-1} * g'(x)."
        return r"Chain rule: if y=g(h(x)) then y' = g'(h(x)) * h'(x). Identify inner h(x)."
    if rules == {"Product"}:
        return r"Product rule: if y=f(x)g(x) then y' = f'(x)g(x) + f(x)g'(x)."
    if rules == {"Quotient"}:
        return r"Quotient rule: if y=\frac{f}{g} then y' = \frac{f'g - fg'}{g^2}."
    if rules == {"Product", "Chain"}:
        return r"Use Product first, then Chain inside parts: y' = f'(x)g(x)+f(x)g'(x), where f or g needs Chain."
    if rules == {"Quotient", "Chain"}:
        return r"Use Quotient, then Chain inside numerator/denominator as needed."
    return "Identify the structure and apply the appropriate rule(s)."

# -----------------------------
# Checking answers
# -----------------------------

def parse_user_expr(s: str) -> sp.Expr:
    return parse_expr(s, transformations=TRANSFORMS, local_dict={"x": x})


def equivalent(expr1: sp.Expr, expr2: sp.Expr) -> bool:
    try:
        return sp.simplify(sp.together(expr1 - expr2)) == 0
    except Exception:
        try:
            return sp.simplify(sp.expand(expr1 - expr2)) == 0
        except Exception:
            return False

# -----------------------------
# UI Helpers
# -----------------------------

def init_state():
    st.session_state.setdefault('score', 0)
    st.session_state.setdefault('streak', 0)
    st.session_state.setdefault('altitude', 0)
    st.session_state.setdefault('best', 0)
    st.session_state.setdefault('lives', 3)
    st.session_state.setdefault('problem', None)
    st.session_state.setdefault('difficulty', 'Beginner')
    st.session_state.setdefault('mode', 'Climb')
    st.session_state.setdefault('seed', None)


def reset_run(seed: int | None = None):
    if seed is not None:
        random.seed(seed)
    st.session_state.update({'score': 0, 'streak': 0, 'altitude': 0, 'lives': 3, 'problem': None})


def altitude_gain(level: int, rule_ok: bool, ans_ok: bool, streak: int) -> int:
    base = {1: 10, 2: 14, 3: 18}[level]
    mult = 1.0
    if rule_ok and ans_ok:
        mult += 0.5
    mult += min(streak, 5)*0.1
    return int(round(base*mult))

# -----------------------------
# App
# -----------------------------

def main():
    st.set_page_config(page_title="Summit of Derivia", page_icon="⛰️", layout="centered")
    init_state()

    with st.sidebar:
        st.title("⛰️ Summit of Derivia")
        st.write("Differentiate your way to the peak. Identify the rule(s) **and** compute the derivative to climb.")
        st.session_state['mode'] = st.selectbox("Mode", ["Climb", "Practice"], index=["Climb", "Practice"].index(st.session_state['mode']))
        st.session_state['difficulty'] = st.radio("Difficulty", ["Beginner", "Intermediate", "Advanced"], index=["Beginner", "Intermediate", "Advanced"].index(st.session_state['difficulty']))
        seed = st.text_input("(Optional) Set random seed for a fixed run")
        if st.button("Start new run"):
            st.session_state['seed'] = int(seed) if seed.strip().isdigit() else None
            reset_run(st.session_state['seed'])
        st.markdown("---")
        st.metric("Score", st.session_state['score'])
        st.metric("Best", st.session_state['best'])
        st.progress(min(1.0, st.session_state['altitude']/1000.0), text=f"Altitude: {st.session_state['altitude']} m / 1000 m")
        st.caption("Reach 1000 m to summit this route.")

    # Retrieve or create current problem
    if st.session_state['problem'] is None:
        st.session_state['problem'] = make_problem(st.session_state['difficulty'])
    problem: Problem = st.session_state['problem']

    colA, colB = st.columns([3,1])
    with colA:
        st.subheader("Current pitch 🧗‍♀️")
        st.latex(sp.latex(sp.Eq(sp.Function('y')(x), problem.expr)))
        st.caption("Select all rules that apply, then enter y'. Use x as the variable. Example inputs: x^2, (x^2+1)^3, (x^2+1)^3*(2x-1)")

        selected = st.multiselect("Which rule(s) apply?", RULE_OPTIONS, default=[])
        user_ans = st.text_input("Enter derivative y' =", key="answer")

        hint_expander = st.expander("Need a hint?")
        with hint_expander:
            st.info(hint_for(problem))

        submit = st.button("Submit")
        if submit:
            rule_ok = set(selected) == problem.rules
            ans_ok = False
            msg_rule = ""
            msg_ans = ""
            try:
                user_expr = parse_user_expr(user_ans)
                correct = sp.diff(problem.expr, x)
                ans_ok = equivalent(sp.simplify(user_expr), sp.simplify(correct))
            except Exception as e:
                msg_ans = f"Parsing error: {e}"

            if rule_ok:
                msg_rule = "✅ Rule(s) correct"
            else:
                msg_rule = f"❌ Rule(s) incorrect. Expected {', '.join(sorted(problem.rules))}"

            if ans_ok:
                msg_ans = "✅ Derivative correct"
            else:
                if not msg_ans:
                    msg_ans = f"❌ Derivative incorrect. Correct y' = {sp.latex(sp.simplify(sp.diff(problem.expr, x)))}"

            st.write("---")
            st.write(msg_rule)
            st.write(msg_ans)

            if st.session_state['mode'] == "Climb":
                if rule_ok and ans_ok:
                    st.session_state['streak'] += 1
                    gain = altitude_gain(problem.level, rule_ok, ans_ok, st.session_state['streak'])
                    st.session_state['altitude'] += gain
                    st.session_state['score'] += 100*problem.level + 10*st.session_state['streak']
                    st.success(f"Nice! You climbed {gain} m. Streak x{st.session_state['streak']}!")
                else:
                    st.session_state['streak'] = 0
                    st.session_state['lives'] -= 1
                    st.error(f"Missed hold! Lives left: {st.session_state['lives']}")
                    if st.session_state['lives'] <= 0:
                        st.warning("Run over. Starting a new route.")
                        st.session_state['best'] = max(st.session_state['best'], st.session_state['score'])
                        reset_run(st.session_state.get('seed'))

                if st.session_state['altitude'] >= 1000:
                    st.balloons()
                    st.success("Summit reached! Route complete. Starting a new route.")
                    st.session_state['best'] = max(st.session_state['best'], st.session_state['score'])
                    reset_run(st.session_state.get('seed'))

            # New problem either way
            st.session_state['problem'] = make_problem(st.session_state['difficulty'])
            st.rerun()

    with colB:
        st.subheader("Stats")
        st.metric("Lives", st.session_state['lives'])
        st.metric("Streak", st.session_state['streak'])
        st.caption("Points scale with difficulty and streak.")

    with st.expander("Teacher notes / controls"):
        st.write("- Beginner: Power + simple Chain; some basic Product/Quotient.\n- Intermediate: Mix of all; occasional combos.\n- Advanced: Frequent Product/Quotient with Chain inside.")
        st.write("- Enter a *seed* in the sidebar to fix the sequence for demonstrations.")
        st.write("- Answers are checked symbolically using SymPy, so equivalent forms are accepted.")


if __name__ == '__main__':
    main()
