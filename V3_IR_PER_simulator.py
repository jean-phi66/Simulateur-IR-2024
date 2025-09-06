import streamlit as st
from streamlit import session_state as ss
import numpy as np
import pandas as pd
import plotly.express as px
from openfisca_core import reforms, periods
from openfisca_france import FranceTaxBenefitSystem
from openfisca_core.simulation_builder import SimulationBuilder

# --- Configuration ---
step = 100
round_num = -2

# --- Helper Functions ---

def round_to(num, round_num=round_num):
    """Rounds a number to the nearest hundred for cleaner display."""
    return int(np.round(num, round_num))

def format_space_thousand_sep(num, trailing=" €"):
    """Formats a number with spaces as thousand separators."""
    return '{:,}'.format(np.round(num, 0)).replace(',', ' ') + trailing

def add_bracket_lines(fig, df_simulation, bareme_annee_simulation):
    """
    Adds vertical lines to the graph at the points where the official TMI bracket changes.
    This method uses the simulation data itself to find the thresholds, ensuring perfect alignment
    with the curve, as it accounts for all complex interactions (10% deduction, CSG, décote, etc.).
    """
    tmi_bracket_changes = df_simulation['ir_tranche'].diff().ne(0)
    change_points = df_simulation[tmi_bracket_changes]

    for _, row in change_points.iterrows():
        bracket_index = int(row['ir_tranche'])
        rbg_threshold = row['Revenu']
        if bracket_index > 0:
            official_rate = bareme_annee_simulation.rates[bracket_index]
            fig.add_vline(x=rbg_threshold, line_width=1, line_dash="dash", line_color="grey", annotation_text=f"TMI {int(official_rate*100)}%", annotation_position="top right", annotation_font_size=10)
    return fig

# --- OpenFisca Reform Definition ---

def modifier_un_parametre(parameters):
    """Applies a custom reform for the 2024 tax year."""
    reform_year = 2024
    reform_period = periods.period(reform_year)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[1].threshold.update(start=reform_period.start, value=11520)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[2].threshold.update(start=reform_period.start, value=29373)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[3].threshold.update(start=reform_period.start, value=83988)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[4].threshold.update(start=reform_period.start, value=180648)
    parameters.impot_revenu.calcul_impot_revenu.plaf_qf.decote.seuil_celib.update(start=reform_period, value=890.)
    parameters.impot_revenu.calcul_impot_revenu.plaf_qf.decote.seuil_couple.update(start=reform_period, value=1473.)
    parameters.impot_revenu.calcul_impot_revenu.plaf_qf.plafond_avantages_procures_par_demi_part.general.update(start=reform_period, value=1794.)
    parameters.impot_revenu.calcul_impot_revenu.plaf_qf.plafond_avantages_procures_par_demi_part.celib_enf.update(start=reform_period, value=4232)
    return parameters

class MaReforme(reforms.Reform):
    """Custom reform class for OpenFisca."""
    def apply(self):
        self.modify_parameters(modifier_function=modifier_un_parametre)

# --- Core Logic: Refactored Functions ---

def create_entities(annee_simulation, nb_adultes, parent_isole, nb_enfants, nb_alternes):
    """Creates the OpenFisca entities dictionary ('CASE')."""
    individus = {f'parent{i+1}': {} for i in range(nb_adultes)}
    individus.update({f'enfant{i+1}': {'enfant_a_charge': {annee_simulation: True}} for i in range(nb_enfants)})
    individus.update({f'enfant{i+1+nb_enfants}': {'garde_alternee': {annee_simulation: True}} for i in range(nb_alternes)})

    foyer_fiscal = {'foyerfiscal1': {
        'declarants': [f'parent{i+1}' for i in range(nb_adultes)],
        'nbH': {annee_simulation: nb_alternes},
        'nb_pac': {annee_simulation: nb_enfants}
    }}
    if nb_adultes == 1 and parent_isole:
        foyer_fiscal['foyerfiscal1']['caseT'] = {annee_simulation: True}

    enfants_all = [f'enfant{i+1}' for i in range(nb_enfants + nb_alternes)]
    case = {
        'individus': individus,
        'foyers_fiscaux': foyer_fiscal,
        'familles': {'famille1': {'parents': [f'parent{i+1}' for i in range(nb_adultes)], 'enfants': enfants_all}},
        'menages': {'menage1': {'personne_de_reference': ['parent1'], 'enfants': enfants_all}}
    }
    if nb_adultes > 1:
        case['menages']['menage1']['conjoint'] = ['parent2']

    n_reshape = nb_adultes + nb_enfants + nb_alternes
    n_reshape_output = 1 + nb_enfants + nb_alternes
    return case, n_reshape, n_reshape_output

def run_simulation(base_case, n_reshape, n_reshape_output, annee_simulation, legislation, axis_variable, axis_max, axis_count):
    """Generic function to run an OpenFisca simulation along one axis."""
    case_sim = base_case.copy()
    case_sim['axes'] = [[{'count': axis_count, 'name': axis_variable, 'min': 0, 'max': axis_max, 'period': annee_simulation}]]
    
    simulation = SimulationBuilder().build_from_entities(legislation, case_sim)
    
    results = {
        'axis_values': simulation.calculate_add(axis_variable, annee_simulation).reshape(axis_count, n_reshape)[:, :len(base_case['foyers_fiscaux']['foyerfiscal1']['declarants'])].sum(axis=1)
    }
    
    variables_to_calculate = ['ip_net', 'avantage_qf', 'ir_taux_marginal', 'ir_tranche', 'taux_moyen_imposition', 'ir_ss_qf', 'decote_gain_fiscal']
    for var in variables_to_calculate:
        # Foyer-level variables are broadcast to all individuals; we reshape and take the first.
        results[var] = simulation.calculate(var, annee_simulation).reshape(axis_count, n_reshape_output)[:, 0]

    # Also get one-shot values that don't vary along the axis
    nb_parts = simulation.calculate('nbptr', annee_simulation)[0]
    
    return results, nb_parts

def run_income_tax_simulation(base_case, n_reshape, n_reshape_output,annee_simulation, salary, salary_max_simu, legislation):
    """Runs the simulation for income tax evolution based on salary."""
    length_simu = round_to(max(salary * 1.2, salary_max_simu))
    length_simu_n = int(length_simu / step)
    
    sim_results, nb_parts = run_simulation(base_case, n_reshape, n_reshape_output, annee_simulation, legislation, 'salaire_imposable', length_simu, length_simu_n)
    
    df = pd.DataFrame({
        'Revenu': sim_results['axis_values'],
        'IR': sim_results['ip_net'],
        'TMI': sim_results['ir_taux_marginal'],
        'Taux moyen d imposition': sim_results['taux_moyen_imposition'],
        'IR sans QF': sim_results['ir_ss_qf'],
        'Reduction QF': -sim_results['avantage_qf'],
        'ir_tranche': sim_results['ir_tranche'],
        'Decote': sim_results['decote_gain_fiscal']
    })
    df['Revenu'] = df['Revenu'].apply(round_to)
    
    bareme = legislation.parameters.impot_revenu.bareme_ir_depuis_1945.bareme(str(annee_simulation))
    return df, bareme, nb_parts, length_simu

def run_per_simulation(base_case, n_reshape, n_reshape_output, annee_simulation, salary, legislation):
    """Runs the simulation for PER payment effect on income tax."""
    base_case['individus']['parent1']['salaire_imposable'] = {annee_simulation: salary}
    length_simu_per = int(salary)
    if length_simu_per == 0:
        return pd.DataFrame(columns=['Versement_PER', 'IR', 'TMI', 'ir_tranche'])

    sim_results, _ = run_simulation(base_case, n_reshape, n_reshape_output,annee_simulation, legislation, 'f6rs', length_simu_per, length_simu_per)

    return pd.DataFrame({
        'Versement_PER': sim_results['axis_values'],
        'IR': sim_results['ip_net'],
        'TMI': sim_results['ir_taux_marginal'],
        'ir_tranche': sim_results['ir_tranche']
    })

def calculate_optimal_per_payment(df_per, df_one_shot, plafond_PER, ir_residuel_min):
    """Calculates the optimal PER payment based on TMI change and residual tax."""
    if df_one_shot.empty or df_per.empty:
        return 0., 0., 0.

    current_tmi_bracket = int(df_one_shot['ir_tranche'].values[0])
    ir_initial = df_one_shot['IR'].values[0]

    # Find the next lower TMI bracket to target
    lower_brackets = df_per['ir_tranche'][df_per['ir_tranche'] < current_tmi_bracket]
    target_bracket = lower_brackets.max() if not lower_brackets.empty else -1

    versement_tmi_opt = 0.
    if target_bracket != -1:
        # Find the smallest payment to reach the target bracket
        df_target_bracket = df_per[df_per['ir_tranche'] == target_bracket]
        if not df_target_bracket.empty:
            versement_tmi_opt = df_target_bracket['Versement_PER'].iloc[0]

    # Filter candidates: must lower the TMI bracket and respect the residual IR
    candidates = df_per[
        (df_per['ir_tranche'] < current_tmi_bracket) &
        (df_per['IR'] >= ir_residuel_min)
    ]

    if not candidates.empty:
        # The optimal payment is the smallest one that meets the criteria
        optimal_row = candidates.iloc[0]
        versement_optimal = optimal_row['Versement_PER']
        impot_final = optimal_row['IR']
    else:
        # If no payment can lower the TMI bracket (or respect residual IR), check if any payment is possible
        # while respecting the residual IR.
        possible_payments = df_per[df_per['IR'] >= ir_residuel_min]
        if not possible_payments.empty:
            # Take the largest possible payment that respects the residual IR
            optimal_row = possible_payments.iloc[-1]
            versement_optimal = optimal_row['Versement_PER']
            impot_final = optimal_row['IR']
        else:
            versement_optimal = 0.
            impot_final = ir_initial

    # The final recommended payment is capped by the user's plafond_PER
    versement_final = min(versement_optimal, plafond_PER)
    
    # Recalculate the final tax if the payment was capped
    if versement_final < versement_optimal and not df_per.empty:
        # Find the closest available payment in the simulation to the capped value
        capped_row = df_per.iloc[(df_per['Versement_PER'] - versement_final).abs().argmin()]
        impot_final = capped_row['IR']

    return versement_final, impot_final, versement_tmi_opt

def run_all_simulations(annee_simulation, nb_adultes, parent_isole, nb_enfants, nb_alternes, salary, plafond_PER, salary_max_simu, ir_residuel_min, legislation):
    """Performs all OpenFisca calculations and returns a dictionary of results."""
    base_case, n_reshape, n_reshape_output = create_entities(annee_simulation, nb_adultes, parent_isole, nb_enfants, nb_alternes)
    
    df_income_tax_evol, bareme, nb_parts, length_simu = run_income_tax_simulation(base_case, n_reshape, n_reshape_output,
                                                                                  annee_simulation, salary, salary_max_simu, legislation)
    
    df_per = run_per_simulation(base_case, n_reshape, n_reshape_output, annee_simulation, salary, legislation)
    
    df_one_shot = df_income_tax_evol[df_income_tax_evol['Revenu'] == round_to(salary)].head(1)
    
    versement_optimal, impot_final, versement_tmi = calculate_optimal_per_payment(df_per, df_one_shot, plafond_PER, ir_residuel_min)

    return {
        "df_income_tax_evol": df_income_tax_evol,
        "df_income_tax_PER": df_per,
        "df_one_shot": df_one_shot,
        "nb_part_one_shot": nb_parts,
        "bareme_annee_simulation": bareme,
        "versement_optimal_PER": versement_optimal,
        "impot_avec_versement": impot_final,
        "versement_PER_TMI": versement_tmi,
        "length_simu": length_simu,
    }
# --- Streamlit UI (Sidebar) ---

#st.set_page_config(layout="wide")
st.title("Simulateur d'Impôt sur le Revenu et d'optimisation PER")

@st.cache_resource
def get_legislation(use_reform):
    """Caches the OpenFisca legislation object."""
    if use_reform:
        legislation_france = FranceTaxBenefitSystem()
        return MaReforme(legislation_france)
    return FranceTaxBenefitSystem()

with st.sidebar:
    st.header("Paramètres du Foyer")
    input_annee_simulation = st.selectbox("Année de perception des revenus", (2024, 2023, 2022, 2021, 2020))
    input_nb_adultes = st.selectbox("Nombre d'adultes", (1, 2))
    input_parent_isole = st.checkbox('Parent isolé ?', disabled=(input_nb_adultes != 1))
    input_nb_enfants = st.selectbox("Enfants à charge", (0, 1, 2, 3, 4, 5))
    input_nb_alternes = st.selectbox("Enfants en garde alternée", (0, 1, 2, 3, 4, 5))

    st.divider()
    st.header("Paramètres de Simulation")
    col1, col2 = st.columns(2)
    input_salary = col1.number_input("Revenu net imposable", value=50000, step=step)
    input_plafond_PER = col2.number_input("Plafond PER", value=16000, step=step)
    input_ir_residuel_min = st.number_input("IR résiduel minimum (pour optimisation)", value=0, step=step, help="L'optimisation du versement PER ne descendra pas l'impôt en dessous de ce montant.")

    with st.expander("Options avancées"):
        input_salary_max_simu = st.number_input("Revenu max. pour les graphiques", value=max(100000, int(input_salary * 1.2)), step=step)
        utilisation_reforme = st.checkbox("Utiliser la réforme PLF 2025 (pour revenus 2024)", value=True, disabled=(input_annee_simulation != 2024))

    button_run = st.button("Lancer la simulation", use_container_width=True, type="primary")

legislation_reforme = get_legislation(utilisation_reforme and input_annee_simulation == 2024)

# --- Display Functions ---

def display_one_shot_tab(results, salary):
    """Displays the content for the 'Calcul de l'IR' tab."""
    df_one_shot = results['df_one_shot']
    if df_one_shot.empty:
        st.error("Impossible de calculer l'impôt pour le revenu spécifié. Essayez d'augmenter le revenu max. pour les graphiques.")
        return

    try:
        IR_one_shot = df_one_shot['IR'].values[0]
        avantage_qf = df_one_shot['Reduction QF'].values[0]
        IR_ss_qf = df_one_shot['IR sans QF'].values[0]
        TMI = df_one_shot['TMI'].values[0]
        taux_moyen = df_one_shot['Taux moyen d imposition'].values[0]
        decote = df_one_shot['Decote'].values[0]

        st.subheader("Synthèse de votre impôt")
        col1, col2, col3 = st.columns(3)
        col1.metric("Revenu net imposable", format_space_thousand_sep(salary))
        col2.metric("Nombre de parts", f"{results['nb_part_one_shot']:.2f}")
        col3.metric("Impôt sur le revenu", format_space_thousand_sep(IR_one_shot))
        
        if IR_ss_qf > 0:
            col1.metric("Effet du quotient familial", format_space_thousand_sep(avantage_qf), f"{np.round(avantage_qf / IR_ss_qf * 100, 1)} %", delta_color='inverse')
        else:
            col1.metric("Effet du quotient familial", format_space_thousand_sep(avantage_qf))

        col2.metric("Tranche Marginale d'Imposition", f"{np.round(TMI * 100, 0):.0f} %")
        col3.metric("Taux moyen d'imposition", f"{np.round(taux_moyen * 100, 1)} %")
        if decote > 0:
            ir_avant_decote = IR_one_shot + decote
            if ir_avant_decote > 0:
                col1.metric("Gain de la décote", format_space_thousand_sep(decote), f"{np.round(decote / ir_avant_decote * 100, 1)} %", delta_color='inverse')
            else:
                col1.metric("Gain de la décote", format_space_thousand_sep(decote))

        st.divider()
        st.subheader("Évolution de l'impôt selon le revenu")
        fig = px.line(results['df_income_tax_evol'], x='Revenu', y='IR', labels={'Revenu': 'Revenu net imposable', 'IR': "Montant de l'IR"})
        add_bracket_lines(fig, results['df_income_tax_evol'], results['bareme_annee_simulation'])
        fig.add_scatter(x=[salary], y=[IR_one_shot], text=format_space_thousand_sep(IR_one_shot), marker=dict(color='red', size=10), name='Votre situation')
        fig.update_layout(xaxis_ticksuffix='€', yaxis_ticksuffix='€', xaxis_range=[0, results['length_simu']])
        st.plotly_chart(fig, use_container_width=True)

    except (IndexError, KeyError) as e:
        st.error(f"Erreur lors de la récupération des données de simulation : {e}")

def display_per_optim_tab(results, salary):
    """Displays the content for the 'Versement Optimal PER' tab."""
    df_one_shot = results['df_one_shot']
    if df_one_shot.empty: return

    IR_one_shot = df_one_shot['IR'].values[0]
    versement_optimal = results['versement_optimal_PER']
    impot_final = results['impot_avec_versement']
    versement_tmi = results['versement_PER_TMI']

    st.subheader("Optimisation du versement PER")
    col1, col2, col3 = st.columns(3)
    col1.metric("Versement PER optimal", format_space_thousand_sep(versement_optimal), help="Versement maximisant la baisse d'impôt, tout en respectant votre plafond PER et l'IR résiduel minimum.")
    col2.metric("Impôt après versement", format_space_thousand_sep(impot_final))
    
    economie = IR_one_shot - impot_final
    if versement_optimal > 0:
        col3.metric("Économie d'impôt", format_space_thousand_sep(economie), f"{np.round(economie / versement_optimal * 100, 1)} % du versement")
    else:
        col3.metric("Économie d'impôt", format_space_thousand_sep(economie))

    if versement_tmi > 0 and versement_tmi <= results.get('plafond_PER', versement_tmi):
        st.info(f"Un versement de **{format_space_thousand_sep(versement_tmi)}** est suffisant pour changer de Tranche Marginale d'Imposition (TMI).", icon="💡")

    st.divider()
    st.subheader("Visualisation de l'optimisation")
    fig = px.line(results['df_income_tax_evol'], x='Revenu', y='IR', labels={'Revenu': 'Revenu net imposable', 'IR': "Montant de l'IR"})
    add_bracket_lines(fig, results['df_income_tax_evol'], results['bareme_annee_simulation'])
    fig.add_scatter(x=[salary], y=[IR_one_shot], text=format_space_thousand_sep(IR_one_shot), marker=dict(color='red', size=10), name='IR initial')
    
    if not results['df_income_tax_evol'].empty:
        equivalent_row = results['df_income_tax_evol'].iloc[(results['df_income_tax_evol']['IR'] - impot_final).abs().argmin()]
        fig.add_scatter(x=[equivalent_row['Revenu']], y=[equivalent_row['IR']], text=format_space_thousand_sep(impot_final), marker=dict(color='green', size=10), name='IR après versement')
    
    fig.update_layout(title="Impact du versement PER optimal sur la courbe d'imposition", xaxis_ticksuffix='€', yaxis_ticksuffix='€', xaxis_range=[0, results['length_simu']])
    st.plotly_chart(fig, use_container_width=True)

def display_per_effect_tab(results, salary, plafond_PER):
    """Displays the content for the 'Effet Versement PER' tab."""
    df_one_shot = results['df_one_shot']
    df_per = results['df_income_tax_PER']
    if df_one_shot.empty or df_per.empty: return

    IR_one_shot = df_one_shot['IR'].values[0]
    versement_input = st.slider("Montant du versement PER", 0, int(plafond_PER), 0, step)
    
    row_selected = df_per.iloc[(df_per['Versement_PER'] - versement_input).abs().argmin()]
    ir_versement = row_selected['IR']
    economie = IR_one_shot - ir_versement
    effort = versement_input - economie

    st.subheader("Impact de votre versement")
    col1, col2, col3 = st.columns(3)
    col1.metric("Impôt après versement", format_space_thousand_sep(ir_versement))
    col2.metric("Économie d'impôt", format_space_thousand_sep(economie))
    col3.metric("Effort d'épargne réel", format_space_thousand_sep(effort), help="Versement moins l'économie d'impôt.")

    st.divider()
    st.subheader("Analyse graphique")
    df_per_filtered = df_per[df_per['Versement_PER'] <= plafond_PER].copy()
    df_per_filtered["Economie_IR"] = IR_one_shot - df_per_filtered['IR']
    
    fig_eco = px.line(df_per_filtered, x='Versement_PER', y='Economie_IR', labels={'Versement_PER': 'Versement PER', 'Economie_IR': "Économie d'impôt"})
    fig_eco.add_scatter(x=[versement_input], y=[economie], text=format_space_thousand_sep(economie), marker=dict(color='green', size=10), name='Votre sélection', showlegend=False)
    fig_eco.update_layout(title="Économie d'impôt en fonction du versement PER", xaxis_ticksuffix='€', yaxis_ticksuffix='€')
    st.plotly_chart(fig_eco, use_container_width=True)

# --- Main App Logic ---

if button_run:
    with st.spinner("Calculs en cours..."):
        ss.simulation_results = run_all_simulations(
            annee_simulation=input_annee_simulation,
            nb_adultes=input_nb_adultes,
            parent_isole=input_parent_isole,
            nb_enfants=input_nb_enfants,
            nb_alternes=input_nb_alternes,
            salary=input_salary,
            plafond_PER=input_plafond_PER,
            salary_max_simu=input_salary_max_simu,
            ir_residuel_min=input_ir_residuel_min,
            legislation=legislation_reforme
        )

if 'simulation_results' in ss:
    results = ss.simulation_results
    tab_one_shot, tab_per_optim, tab_per_effect = st.tabs(["Calcul de l'IR", "Versement Optimal PER", "Effet Versement PER"])

    with tab_one_shot:
        display_one_shot_tab(results, input_salary)

    with tab_per_optim:
        display_per_optim_tab(results, input_salary)

    with tab_per_effect:
        display_per_effect_tab(results, input_salary, input_plafond_PER)
