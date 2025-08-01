import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from flask import jsonify # Added for custom error handler
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu, sem, t as t_dist # aliased t to t_dist
# from flask_talisman import Talisman # Alternative: Manually set headers

# --- 0. Load Data from CSV on disk ---
csv_path = os.path.join(os.getcwd(), "enriched_metabolite_data.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    print(f"WARNING: {csv_path} not found. Using empty DataFrame instead.")
    df = pd.DataFrame()


# --- Global App Instantiation ---
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}])
server = app.server
# --- Security Headers via Flask-Talisman ---
# Talisman(server, force_https=False) # Original Talisman line

# Alternative: Manually set common security headers
@server.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN' # Or 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block' # Older header, CSP is preferred
    # Content-Security-Policy: A basic policy. You'll likely need to refine this
    # based on your app's specific needs (e.g., if you use external fonts, scripts, or connect to other domains).
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; object-src 'none'; frame-ancestors 'self'; worker-src 'self' blob:;"
    # Referrer-Policy: Controls how much referrer information is sent.
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Permissions-Policy: Controls browser features. Disable features not needed by the dashboard.
    # Example: Disable microphone, camera, geolocation. Add others as needed.
    response.headers['Permissions-Policy'] = "microphone=(), camera=(), geolocation=(), payment=(), usb=(), autoplay=()"
    # Strict-Transport-Security (HSTS): Enforces HTTPS.
    # Only add HSTS if your site is consistently served over HTTPS and not in debug mode.
    if not server.debug: # server is app.server, app.debug is False in production
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains' # 1 year
    # For more complex CSP/HSTS, consider Flask-Talisman or reverse proxy (e.g., Nginx) configuration.
    return response

# --- Custom Error Handlers ---
@server.errorhandler(405)
def method_not_allowed(e):
    """Custom 405 Method Not Allowed error handler."""
    return jsonify(error=str(e), message="The method is not allowed for the requested URL."), 405

@server.errorhandler(500)
def internal_server_error(e):
    """Custom 500 Internal Server Error handler."""
    import traceback
    tb_str = traceback.format_exc()
    # Using print for consistency with other error logging in this file.
    # In a production app, app.logger.error (after configuring Flask logging) would be preferred.    
    print(f"Unhandled Internal Server Error: {e}\nTraceback:\n{tb_str}")
    
    error_str = "Internal Server Error" # Default
    message_str = "An unexpected error occurred on the server. Please try again later or contact support if the issue persists."
    
    try:
        error_val_str = str(e)
        # Limit length to prevent issues with overly long error strings if 'e' is unusual or contains sensitive info
        error_str = error_val_str[:1000] + "..." if len(error_val_str) > 1000 else error_val_str
    except Exception as e_str_err:
        print(f"CRITICAL: Error converting original exception to string in 500 handler: {e_str_err}")
        # error_str remains "Internal Server Error"
        
    return jsonify(
        error=error_str,
        message=message_str
    ), 500

cyto.load_extra_layouts()

# --- Sticky Note Style ---
sticky_note_style = {
    'backgroundColor': '#feffc0',
    'borderLeft': '5px solid #f9d71c',
    'padding': '12px',
    'marginTop': '8px',
    'marginBottom': '15px',
    'borderRadius': '4px',
    'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)',
    'fontSize': '0.9em',
    'lineHeight': '1.4',
}
sticky_note_title_style = {
    'marginTop': '0',
    'marginBottom': '5px',
    'color': '#b8860b',
    'fontWeight': 'bold'
}

# --- Global Data Preparations & Constants ---
P_VALUE_THRESHOLD_GLOBAL = 0.05
TOP_N_GLOBAL = 5

# Potential chemical descriptors for the new profile plot
# These will be filtered to include only numeric columns present in the loaded DataFrame
POTENTIAL_CHEM_DESCRIPTORS = sorted(list(set([
    'qm_total_energy', 'qm_homo', 'qm_lumo', 'qm_gap', 'qm_dipole_moment',
    'xlogp', 'fsp3', 'complexity',
    'hbond_donors', 'hbond_acceptors', 'tpsa',
    'rotatable_bonds', 'mol_weight_da'
])))

if 'direction_flag' in df.columns and df['direction_flag'].dtype == 'object':
    df['direction_flag'] = df['direction_flag'].str.strip("'")

# Ensure unique DFs are created safely even if original df is minimal
df_metabolites_unique_global_overview = df.drop_duplicates(subset=['metabolite_name']).copy() if 'metabolite_name' in df.columns else pd.DataFrame()
df_genes_unique_global_overview = df.drop_duplicates(subset=['gene_symbol']).copy() if 'gene_symbol' in df.columns else pd.DataFrame()
df_tfs_unique_global_overview = df.drop_duplicates(subset=['transcription_factor']).copy() if 'transcription_factor' in df.columns else pd.DataFrame()


# Volcano Plots Data for Global Overview (with safety checks)
required_met_cols_volcano = ['metabolite_pval', 'metabolite_log2fc']
if not df_metabolites_unique_global_overview.empty and all(col in df_metabolites_unique_global_overview for col in required_met_cols_volcano):
    df_metabolites_unique_global_overview.loc[:, '-log10p_metabolite'] = -np.log10(df_metabolites_unique_global_overview['metabolite_pval'].astype(float).replace(0, 1e-300))
    finite_logp_met = df_metabolites_unique_global_overview['-log10p_metabolite'][np.isfinite(df_metabolites_unique_global_overview['-log10p_metabolite'])]
    max_finite_logp_met = np.nanmax(finite_logp_met) if not finite_logp_met.empty else 0
    df_metabolites_unique_global_overview.loc[:, '-log10p_metabolite'] = df_metabolites_unique_global_overview['-log10p_metabolite'].replace([np.inf, -np.inf], max_finite_logp_met)
    df_metabolites_unique_global_overview.loc[:, 'significant_metabolite'] = (df_metabolites_unique_global_overview['metabolite_pval'] < P_VALUE_THRESHOLD_GLOBAL) & (df_metabolites_unique_global_overview['metabolite_log2fc'].abs() > 1)
else:
    if not df_metabolites_unique_global_overview.empty: # Add columns if df exists but cols are missing
        df_metabolites_unique_global_overview['-log10p_metabolite'] = pd.Series(dtype=float)
        df_metabolites_unique_global_overview['significant_metabolite'] = pd.Series(dtype=bool)

required_gene_cols_volcano = ['gene_expr_pval', 'gene_log2fc']
if not df_genes_unique_global_overview.empty and all(col in df_genes_unique_global_overview for col in required_gene_cols_volcano):
    df_genes_unique_global_overview.loc[:, '-log10p_gene'] = -np.log10(df_genes_unique_global_overview['gene_expr_pval'].astype(float).replace(0, 1e-300))
    finite_logp_gene = df_genes_unique_global_overview['-log10p_gene'][np.isfinite(df_genes_unique_global_overview['-log10p_gene'])]
    max_finite_logp_gene = np.nanmax(finite_logp_gene) if not finite_logp_gene.empty else 0
    df_genes_unique_global_overview.loc[:, '-log10p_gene'] = df_genes_unique_global_overview['-log10p_gene'].replace([np.inf, -np.inf], max_finite_logp_gene)
    df_genes_unique_global_overview.loc[:, 'significant_gene'] = (df_genes_unique_global_overview['gene_expr_pval'] < P_VALUE_THRESHOLD_GLOBAL) & (df_genes_unique_global_overview['gene_log2fc'].abs() > 0.5)
else:
    if not df_genes_unique_global_overview.empty:
        df_genes_unique_global_overview['-log10p_gene'] = pd.Series(dtype=float)
        df_genes_unique_global_overview['significant_gene'] = pd.Series(dtype=bool)


# Summary Statistics for Global Overview
num_unique_metabolites_global = df['metabolite_name'].nunique() if 'metabolite_name' in df.columns else 0
num_unique_genes_global = df['gene_symbol'].nunique() if 'gene_symbol' in df.columns else 0
num_unique_kegg_pathways_global = df['kegg_pathway_name'].nunique() if 'kegg_pathway_name' in df.columns else 0
num_unique_tfs_global = df['transcription_factor'].nunique() if 'transcription_factor' in df.columns else 0

significant_metabolites_count_global = 0
if not df_metabolites_unique_global_overview.empty and 'significant_metabolite' in df_metabolites_unique_global_overview.columns:
    significant_metabolites_count_global = df_metabolites_unique_global_overview['significant_metabolite'].sum()

significant_genes_count_global = 0
if not df_genes_unique_global_overview.empty and 'significant_gene' in df_genes_unique_global_overview.columns:
    significant_genes_count_global = df_genes_unique_global_overview['significant_gene'].sum()

significant_tfs_count_global = 0
if not df_tfs_unique_global_overview.empty and 'tf_activity_pval' in df_tfs_unique_global_overview.columns and 'transcription_factor' in df_tfs_unique_global_overview.columns:
    significant_tfs_count_global = df_tfs_unique_global_overview[df_tfs_unique_global_overview['tf_activity_pval'] < P_VALUE_THRESHOLD_GLOBAL]['transcription_factor'].nunique()


# Distribution Plots Data for Global Overview
metabolite_log2fc_dist_global = df_metabolites_unique_global_overview['metabolite_log2fc'].dropna() if not df_metabolites_unique_global_overview.empty and 'metabolite_log2fc' in df_metabolites_unique_global_overview else pd.Series(dtype=float)
gene_log2fc_dist_global = df_genes_unique_global_overview['gene_log2fc'].dropna() if not df_genes_unique_global_overview.empty and 'gene_log2fc' in df_genes_unique_global_overview else pd.Series(dtype=float)
tf_activity_score_dist_global = df_tfs_unique_global_overview['tf_activity_score'].dropna() if not df_tfs_unique_global_overview.empty and 'tf_activity_score' in df_tfs_unique_global_overview else pd.Series(dtype=float)
direction_counts_global = df_metabolites_unique_global_overview['direction_flag'].value_counts() if not df_metabolites_unique_global_overview.empty and 'direction_flag' in df_metabolites_unique_global_overview else pd.Series(dtype=int)


# Top Hits for Global Overview
top_hits_req_met_cols = ['metabolite_pval', 'metabolite_log2fc', 'metabolite_name']
if not df_metabolites_unique_global_overview.empty and all(col in df_metabolites_unique_global_overview for col in top_hits_req_met_cols):
    significant_met_df_global = df_metabolites_unique_global_overview[df_metabolites_unique_global_overview['metabolite_pval'] < P_VALUE_THRESHOLD_GLOBAL].copy()
    if not significant_met_df_global.empty :
        significant_met_df_global.loc[:, 'abs_log2fc'] = significant_met_df_global['metabolite_log2fc'].abs()
        top_up_metabolites_global = significant_met_df_global.sort_values(by='metabolite_log2fc', ascending=False).head(TOP_N_GLOBAL)
        top_down_metabolites_global = significant_met_df_global.sort_values(by='metabolite_log2fc', ascending=True).head(TOP_N_GLOBAL)
        top_bottom_metabolites_df_global = pd.concat([top_up_metabolites_global, top_down_metabolites_global.iloc[::-1]]).drop_duplicates(subset=['metabolite_name'])
    else:
        top_bottom_metabolites_df_global = pd.DataFrame(columns=['metabolite_name', 'metabolite_log2fc'])
else:
    top_bottom_metabolites_df_global = pd.DataFrame(columns=['metabolite_name', 'metabolite_log2fc'])

top_hits_req_gene_cols = ['gene_expr_pval', 'gene_log2fc', 'gene_symbol']
if not df_genes_unique_global_overview.empty and all(col in df_genes_unique_global_overview for col in top_hits_req_gene_cols):
    significant_gene_df_global = df_genes_unique_global_overview[df_genes_unique_global_overview['gene_expr_pval'] < P_VALUE_THRESHOLD_GLOBAL].copy()
    if not significant_gene_df_global.empty:
        top_up_genes_global = significant_gene_df_global.sort_values(by='gene_log2fc', ascending=False).head(TOP_N_GLOBAL)
        top_down_genes_global = significant_gene_df_global.sort_values(by='gene_log2fc', ascending=True).head(TOP_N_GLOBAL)
        top_bottom_genes_df_global = pd.concat([top_up_genes_global, top_down_genes_global.iloc[::-1]]).drop_duplicates(subset=['gene_symbol'])
    else:
        top_bottom_genes_df_global = pd.DataFrame(columns=['gene_symbol', 'gene_log2fc'])
else:
    top_bottom_genes_df_global = pd.DataFrame(columns=['gene_symbol', 'gene_log2fc'])

top_hits_req_tf_cols = ['tf_activity_pval', 'tf_activity_score', 'transcription_factor']
if not df_tfs_unique_global_overview.empty and all(col in df_tfs_unique_global_overview for col in top_hits_req_tf_cols):
    significant_tf_df_global = df_tfs_unique_global_overview[df_tfs_unique_global_overview['tf_activity_pval'] < P_VALUE_THRESHOLD_GLOBAL].copy()
    if not significant_tf_df_global.empty:
        significant_tf_df_global.loc[:, 'abs_activity_score'] = significant_tf_df_global['tf_activity_score'].abs()
        top_active_tfs_global = significant_tf_df_global.sort_values(by='tf_activity_score', ascending=False).head(TOP_N_GLOBAL)
        top_inactive_tfs_global = significant_tf_df_global.sort_values(by='tf_activity_score', ascending=True).head(TOP_N_GLOBAL)
        top_tfs_df_global = pd.concat([top_active_tfs_global, top_inactive_tfs_global.iloc[::-1]]).drop_duplicates(subset=['transcription_factor'])
    else:
        top_tfs_df_global = pd.DataFrame(columns=['transcription_factor', 'tf_activity_score'])
else:
    top_tfs_df_global = pd.DataFrame(columns=['transcription_factor', 'tf_activity_score'])

# --- Top Enriched Pathways for Global Overview ---
df_top_enriched_pathways_global = pd.DataFrame(columns=['kegg_pathway_name', 'pathway_score'])
TOP_N_PATHWAYS_GLOBAL = 10 # Define how many top pathways to show

# Use the main df for this, as pathway_score might not be on a "unique pathway" pre-filtered df.
if not df.empty and 'kegg_pathway_name' in df.columns and 'pathway_score' in df.columns:
    # Create a temporary DataFrame with only the necessary columns
    pathways_df_temp = df[['kegg_pathway_name', 'pathway_score']].copy()
    
    # Convert pathway_score to numeric, coercing errors, and drop rows where either is NaN
    pathways_df_temp['pathway_score'] = pd.to_numeric(pathways_df_temp['pathway_score'], errors='coerce')
    pathways_df_temp.dropna(subset=['kegg_pathway_name', 'pathway_score'], inplace=True)
    
    if not pathways_df_temp.empty:
        # Sort by pathway_score descending, then drop duplicates of kegg_pathway_name, keeping the highest score.
        # This ensures each pathway is represented once with its maximum score if multiple scores exist for the same pathway name.
        df_top_enriched_pathways_global = pathways_df_temp.sort_values(by='pathway_score', ascending=False)\
                                                          .drop_duplicates(subset=['kegg_pathway_name'], keep='first')\
                                                          .head(TOP_N_PATHWAYS_GLOBAL)



# Figures for Global Overview Tab
empty_figure_layout = lambda title: go.Figure().update_layout(title_text=title + " (No data available)", annotations=[dict(text="No data to display", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

# Dynamically determine nbins for histograms
def get_optimal_nbins(data_series, max_bins=30, min_points_for_max_bins=50):
    if data_series.empty:
        return 10 # Default for empty or very small series
    n_unique_points = data_series.nunique()
    if n_unique_points <= 1: # Handle single point or no variation
        return 1
    if n_unique_points < 10: # For very few unique points, use fewer bins
        return max(1, n_unique_points // 2)
    
    # Sturges' formula as a guideline, capped by max_bins and n_unique_points
    sturges_bins = int(np.ceil(1 + np.log2(n_unique_points)))
    
    # If many points but few unique values, don't over-bin
    return min(sturges_bins, max_bins, n_unique_points)

fig_met_dist_global = px.histogram(metabolite_log2fc_dist_global, nbins=get_optimal_nbins(metabolite_log2fc_dist_global, max_bins=20), title="Metabolite log2FC Distribution") if not metabolite_log2fc_dist_global.empty else empty_figure_layout("Metabolite log2FC Distribution")
fig_gene_dist_global = px.histogram(gene_log2fc_dist_global, nbins=get_optimal_nbins(gene_log2fc_dist_global), title="Gene log2FC Distribution") if not gene_log2fc_dist_global.empty else empty_figure_layout("Gene log2FC Distribution")
fig_tf_dist_global = px.histogram(tf_activity_score_dist_global, nbins=get_optimal_nbins(tf_activity_score_dist_global, max_bins=25), title="TF Activity Score Distribution") if not tf_activity_score_dist_global.empty else empty_figure_layout("TF Activity Score Distribution")
fig_direction_pie_global = px.pie(values=direction_counts_global.values, names=direction_counts_global.index, title="Metabolite Regulation Direction") if not direction_counts_global.empty else empty_figure_layout("Metabolite Regulation Direction")

fig_top_met_global = px.bar(top_bottom_metabolites_df_global, y='metabolite_name', x='metabolite_log2fc', orientation='h', title=f"Top Up/Down-Regulated Metabolites (p < {P_VALUE_THRESHOLD_GLOBAL})", color='metabolite_log2fc', color_continuous_scale=px.colors.diverging.RdBu_r, labels={'metabolite_log2fc': 'log2 Fold Change', 'metabolite_name': 'Metabolite'}) if not top_bottom_metabolites_df_global.empty else empty_figure_layout(f"Top Up/Down-Regulated Metabolites")
if not top_bottom_metabolites_df_global.empty: fig_top_met_global.update_layout(yaxis={'categoryorder':'total ascending'})

fig_top_gene_global = px.bar(top_bottom_genes_df_global, y='gene_symbol', x='gene_log2fc', orientation='h', title=f"Top Up/Down-Regulated Genes (p < {P_VALUE_THRESHOLD_GLOBAL})", color='gene_log2fc', color_continuous_scale=px.colors.diverging.RdBu_r, labels={'gene_log2fc': 'log2 Fold Change', 'gene_symbol': 'Gene Symbol'}) if not top_bottom_genes_df_global.empty else empty_figure_layout(f"Top Up/Down-Regulated Genes")
if not top_bottom_genes_df_global.empty: fig_top_gene_global.update_layout(yaxis={'categoryorder':'total ascending'})

fig_top_tf_global = px.bar(top_tfs_df_global, y='transcription_factor', x='tf_activity_score', orientation='h', title=f"Top TFs by Activity Score (p < {P_VALUE_THRESHOLD_GLOBAL})", color='tf_activity_score', color_continuous_scale=px.colors.diverging.Portland_r, labels={'tf_activity_score': 'Activity Score', 'transcription_factor': 'Transcription Factor'}) if not top_tfs_df_global.empty else empty_figure_layout(f"Top TFs by Activity Score")
if not top_tfs_df_global.empty: fig_top_tf_global.update_layout(yaxis={'categoryorder':'total ascending'})

fig_top_enriched_pathways_global = empty_figure_layout(f"Top {TOP_N_PATHWAYS_GLOBAL} Enriched Pathways by Score")
if not df_top_enriched_pathways_global.empty:
    fig_top_enriched_pathways_global = px.bar(
        df_top_enriched_pathways_global,
        y='kegg_pathway_name',
        x='pathway_score',
        orientation='h',
        title=f"Top {TOP_N_PATHWAYS_GLOBAL} Enriched Pathways by Score",
        labels={'pathway_score': 'Pathway Score', 'kegg_pathway_name': 'KEGG Pathway Name'},
        color='pathway_score',
        color_continuous_scale=px.colors.sequential.Viridis  # Using a Viridis-like sequential scale
    )
    fig_top_enriched_pathways_global.update_layout(yaxis={'categoryorder':'total ascending'})

fig_volcano_met_global = px.scatter(df_metabolites_unique_global_overview, x='metabolite_log2fc', y='-log10p_metabolite', hover_data=['metabolite_name', 'metabolite_pval', 'vip_score'], color='significant_metabolite', color_discrete_map={True: 'red', False: 'blue'}, title="Metabolite Volcano Plot", labels={'metabolite_log2fc': 'log2 Fold Change', '-log10p_metabolite': '-log10 (p-value)'}) if not df_metabolites_unique_global_overview.empty and '-log10p_metabolite' in df_metabolites_unique_global_overview.columns else empty_figure_layout("Metabolite Volcano Plot")
if not df_metabolites_unique_global_overview.empty and '-log10p_metabolite' in df_metabolites_unique_global_overview.columns:
    fig_volcano_met_global.add_hline(y=-np.log10(P_VALUE_THRESHOLD_GLOBAL), line_dash="dash", line_color="grey")
    fig_volcano_met_global.add_vline(x=1, line_dash="dash", line_color="grey")
    fig_volcano_met_global.add_vline(x=-1, line_dash="dash", line_color="grey")

fig_volcano_gene_global = px.scatter(df_genes_unique_global_overview, x='gene_log2fc', y='-log10p_gene', hover_data=['gene_symbol', 'gene_expr_pval'], color='significant_gene', color_discrete_map={True: 'red', False: 'blue'}, title="Gene Volcano Plot", labels={'gene_log2fc': 'log2 Fold Change', '-log10p_gene': '-log10 (p-value)'}) if not df_genes_unique_global_overview.empty and '-log10p_gene' in df_genes_unique_global_overview.columns else empty_figure_layout("Gene Volcano Plot")
if not df_genes_unique_global_overview.empty and '-log10p_gene' in df_genes_unique_global_overview.columns:
    fig_volcano_gene_global.add_hline(y=-np.log10(P_VALUE_THRESHOLD_GLOBAL), line_dash="dash", line_color="grey")
    fig_volcano_gene_global.add_vline(x=0.5, line_dash="dash", line_color="grey")
    fig_volcano_gene_global.add_vline(x=-0.5, line_dash="dash", line_color="grey")

# --- Plot Generation Functions for Global Overview ---
def generate_chem_descriptor_profile_figure(df_met_unique, potential_descriptors_list):
    plot_title = "Chemical Descriptor Profiles by Regulation"
    if df_met_unique.empty or 'metabolite_log2fc' not in df_met_unique.columns:
        return empty_figure_layout(plot_title)

    # Filter for relevant data and define direction_flag
    df_plot = df_met_unique[['metabolite_name', 'metabolite_log2fc'] + potential_descriptors_list].copy()
    df_plot.dropna(subset=['metabolite_log2fc'], inplace=True)
    if df_plot.empty:
        return empty_figure_layout(plot_title)
        
    df_plot['direction_flag'] = np.where(df_plot['metabolite_log2fc'] > 0, 'up', 'down')

    # Identify actual numeric descriptor columns present in the data
    actual_descriptors = [
        col for col in potential_descriptors_list 
        if col in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[col])
    ]
    if not actual_descriptors:
        return empty_figure_layout(f"{plot_title} (No numeric descriptors found)")

    # Drop rows with NaNs in any of the actual_descriptors for modeling and plotting
    df_plot.dropna(subset=actual_descriptors, inplace=True)
    if df_plot.shape[0] < 2 or df_plot['direction_flag'].nunique() < 2: # Need at least 2 samples and 2 groups
        return empty_figure_layout(f"{plot_title} (Insufficient data for comparison)")

    X = df_plot[actual_descriptors]
    y = df_plot['direction_flag']
    
    # Feature importance and ordering
    ordered_descriptors = actual_descriptors # Default order
    smallest_class_size = y.value_counts().min()
    
    if smallest_class_size >= 2 and X.shape[0] >= 4 : # Min 2 samples per class for n_splits=2
        try:
            # Ensure n_splits is at most the size of the smallest class, but at least 2
            n_cv_splits = 2 # As per user request: min(2, smallest_class_size) means 2 if smallest_class_size >=2
            
            cv = RepeatedStratifiedKFold(n_splits=n_cv_splits, n_repeats=3, random_state=42)
            model = RandomForestClassifier(random_state=42, n_estimators=50, class_weight='balanced') # Reduced n_estimators for speed
            
            importances_sum = np.zeros(X.shape[1])
            num_successful_fits = 0
            for train_idx, test_idx in cv.split(X, y):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                if len(np.unique(y_train)) > 1: # Ensure training fold has more than one class
                    model.fit(X_train, y_train)
                    importances_sum += model.feature_importances_
                    num_successful_fits +=1
            
            if num_successful_fits > 0:
                avg_importances = importances_sum / num_successful_fits
                ordered_descriptors = [desc for _, desc in sorted(zip(avg_importances, actual_descriptors), reverse=True)]
            else: # Fallback if CV fitting failed (e.g. all folds had single class)
                print("Warning: RF for feature importance had no successful fits. Using alphabetical order for descriptors.")
                ordered_descriptors = sorted(actual_descriptors)
        except Exception as e: # Catch any other sklearn errors
            print(f"Warning: RF for feature importance failed: {e}. Using alphabetical order for descriptors.")
            ordered_descriptors = sorted(actual_descriptors)
    else:
        print("Warning: Not enough samples per class for CV. Using alphabetical order for descriptors.")
        ordered_descriptors = sorted(actual_descriptors)

    # Standardization
    scaler = StandardScaler()
    df_plot[ordered_descriptors] = scaler.fit_transform(df_plot[ordered_descriptors])

    # Melt DataFrame
    df_melted = df_plot.melt(id_vars=['metabolite_name', 'direction_flag'], value_vars=ordered_descriptors, 
                             var_name='descriptor', value_name='z_score')

    fig = go.Figure()
    colors = {'up': '#CD5C5C', 'down': '#6495ED'} # Use hex codes for indianred and cornflowerblue
    TOP_N_DIFFERENCES_PROFILE = 3

    # Alternating background
    for i, desc_name in enumerate(ordered_descriptors):
        if i % 2 == 1:
            fig.add_vrect(x0=i-0.5, x1=i+0.5, fillcolor="rgba(0,0,0,0.03)", layer="below", line_width=0)

    # Individual faint traces
    for _, group_df in df_melted.groupby('metabolite_name'):
        fig.add_trace(go.Scatter(x=group_df['descriptor'], y=group_df['z_score'], mode='lines',
                                 line=dict(width=0.5, color='rgba(128,128,128,0.3)'), showlegend=False,
                                 hoverinfo='skip'))
    
    descriptor_stats = {} # To store means for top diff calculation and p-values
    for group_label, group_color in colors.items():
        group_data = df_melted[df_melted['direction_flag'] == group_label]
        if group_data.empty: continue

        summary = group_data.groupby('descriptor')['z_score'].agg(['mean', 'sem', 'count']).reindex(ordered_descriptors)
        summary['ci_95_hi'] = summary['mean'] + summary['sem'] * t_dist.ppf((1 + 0.95) / 2., summary['count'] - 1)
        summary['ci_95_lo'] = summary['mean'] - summary['sem'] * t_dist.ppf((1 + 0.95) / 2., summary['count'] - 1)
        
        # Store means for later
        for desc_name, row_data in summary.iterrows():
            if desc_name not in descriptor_stats: descriptor_stats[desc_name] = {}
            descriptor_stats[desc_name][group_label + '_mean'] = row_data['mean']

        fig.add_trace(go.Scatter(x=summary.index, y=summary['ci_95_hi'], mode='lines', line_color=group_color, line_width=0, showlegend=False))
        fig.add_trace(go.Scatter(x=summary.index, y=summary['ci_95_lo'], mode='lines', line_color=group_color, line_width=0, fill='tonexty', fillcolor=f'rgba({int(group_color[1:3],16)},{int(group_color[3:5],16)},{int(group_color[5:7],16)},0.1)', showlegend=False, name=f'{group_label} 95% CI'))
        fig.add_trace(go.Scatter(x=summary.index, y=summary['mean'], mode='lines+markers', name=f'Mean {group_label}-regulated', line_color=group_color, marker_size=5))

    # Mann-Whitney U and significance stars
    max_y_val_for_stars = df_melted['z_score'].max() if not df_melted.empty else 0
    for i, desc_name in enumerate(ordered_descriptors):
        up_scores = df_plot[df_plot['direction_flag'] == 'up'][desc_name].dropna()
        down_scores = df_plot[df_plot['direction_flag'] == 'down'][desc_name].dropna()
        if len(up_scores) > 1 and len(down_scores) > 1: # Need at least 2 per group for test
            _, p_val = mannwhitneyu(up_scores, down_scores, alternative='two-sided')
            if desc_name in descriptor_stats: descriptor_stats[desc_name]['p_value'] = p_val
            if p_val < P_VALUE_THRESHOLD_GLOBAL:
                fig.add_annotation(x=desc_name, y=max_y_val_for_stars + 0.2, text="*", showarrow=False, font=dict(size=16, color="black"))

    # Top 3 differences callouts
    diff_df = pd.DataFrame.from_dict(descriptor_stats, orient='index')
    if 'up_mean' in diff_df.columns and 'down_mean' in diff_df.columns:
        diff_df['abs_diff'] = (diff_df['up_mean'] - diff_df['down_mean']).abs()
        top_diff_descriptors = diff_df.nlargest(TOP_N_DIFFERENCES_PROFILE, 'abs_diff').index
        for desc_name in top_diff_descriptors:
            # Determine y-position for callout, ensuring it's within a reasonable plot range if y-axis is clipped
            # For now, base it on actual data, clipping will handle visibility.
            y_val_up = diff_df.loc[desc_name, 'up_mean'] if 'up_mean' in diff_df.columns and pd.notna(diff_df.loc[desc_name, 'up_mean']) else -np.inf
            y_val_down = diff_df.loc[desc_name, 'down_mean'] if 'down_mean' in diff_df.columns and pd.notna(diff_df.loc[desc_name, 'down_mean']) else -np.inf
            y_pos_callout_base = max(y_val_up, y_val_down)
            y_pos_callout = y_pos_callout_base + 0.5 if y_pos_callout_base != -np.inf else max_y_val_for_stars # Fallback if means are missing

            fig.add_annotation(x=desc_name, y=y_pos_callout, text=desc_name, showarrow=True, arrowhead=1, 
                               bgcolor="yellow", bordercolor="black", borderwidth=1, ax=0, ay=-30,
                               font=dict(color="black")) # Black text for callout

    # Add horizontal lines for Z-score reference
    for z_val in [-2, -1, 0, 1, 2]:
        fig.add_hline(y=z_val, line_dash="dot", line_color="grey", opacity=0.7, layer="below")

    fig.update_layout(title=plot_title, xaxis_title="Descriptor (Ordered by Cross-Validated Importance)", yaxis_title="Z-score (Mean with 95% CI)", yaxis_range=[-3.5, 3.5], # Constrain Y-axis
                      xaxis_categoryorder='array', xaxis_categoryarray=ordered_descriptors, legend_title_text='Regulation', height=600,
                      hovermode="x unified")
    return fig

def generate_pathway_descriptor_heatmap_figure(df_main, top_pathways_names_list, potential_descriptors_list):
    plot_title = "Pathway-Level Mean Z-scores of Chemical Descriptors"

    if df_main.empty or 'kegg_pathway_name' not in df_main.columns or not top_pathways_names_list:
        return empty_figure_layout(f"{plot_title} (Insufficient data for pathways or descriptors)")

    actual_descriptors = [
        col for col in potential_descriptors_list
        if col in df_main.columns and pd.api.types.is_numeric_dtype(df_main[col])
    ]
    if not actual_descriptors:
        return empty_figure_layout(f"{plot_title} (No numeric chemical descriptors found)")

    # Prepare data: select necessary columns and drop rows with NaN in any actual_descriptor
    df_heatmap_prep = df_main[['metabolite_name', 'kegg_pathway_name'] + actual_descriptors].copy()
    df_heatmap_prep.dropna(subset=actual_descriptors, inplace=True)
    if df_heatmap_prep.empty:
        return empty_figure_layout(f"{plot_title} (No data after NaN drop in descriptors)")

    # Z-score normalize the descriptors across the metabolites that have these values
    scaler = StandardScaler()
    df_heatmap_prep[actual_descriptors] = scaler.fit_transform(df_heatmap_prep[actual_descriptors])

    # Filter for metabolites/entries that are part of the top pathways
    df_filtered_for_top_pathways = df_heatmap_prep[df_heatmap_prep['kegg_pathway_name'].isin(top_pathways_names_list)].copy()
    if df_filtered_for_top_pathways.empty:
        return empty_figure_layout(f"{plot_title} (No metabolites found in top pathways with descriptor data)")

    # Compute mean Z-score of each descriptor for each pathway
    heatmap_data_means = df_filtered_for_top_pathways.groupby('kegg_pathway_name')[actual_descriptors].mean()

    # Reorder rows (pathways) to match top_pathways_names_list and ensure only these are included.
    # Also, drop any pathways from top_pathways_names_list that ended up with no data (all NaNs after mean).
    # And drop descriptor columns if they are all NaN across the selected pathways.
    heatmap_data_final = heatmap_data_means.reindex(index=top_pathways_names_list, columns=actual_descriptors).dropna(how='all', axis=0).dropna(how='all', axis=1)

    if heatmap_data_final.empty:
         return empty_figure_layout(f"{plot_title} (No data for heatmap after final processing)")

    # Determine symmetrical color scale limits for Z-scores
    min_val = heatmap_data_final.min().min()
    max_val = heatmap_data_final.max().max()
    z_abs_max = 1.0 # Default if all values are NaN or zero
    if pd.notna(min_val) and pd.notna(max_val) :
        z_abs_max = max(abs(min_val), abs(max_val), 0.1) # Use 0.1 to avoid zero range if all data is 0

    fig = px.imshow(
        heatmap_data_final,
        labels=dict(x="Chemical Descriptor", y="KEGG Pathway", color="Mean Z-score"),
        color_continuous_scale='RdBu_r', # Diverging scale, good for Z-scores
        color_continuous_midpoint=0,    # Center the color scale at 0
        zmin=-z_abs_max,
        zmax=z_abs_max,
        aspect="auto"
    )
    fig.update_layout(
        title_text=plot_title,
        xaxis_title="Chemical Descriptor",
        yaxis_title="KEGG Pathway (Ordered by Score)",
        xaxis_tickangle=-45,
        height=max(450, 35 * len(heatmap_data_final.index) + 200), # Dynamic height
        margin=dict(l=350, r=50, t=80, b=150) # Adjust margins for labels
    )
    return fig

def generate_tf_target_chem_fingerprints_figure(df_main, df_metabolites_unique, potential_descriptors_list, p_value_thresh, top_n_tfs_each_dir):
    plot_title = f"Chemical Fingerprints of Metabolites Associated with Top {top_n_tfs_each_dir} Active/Repressed TFs"

    # 1. Identify actual numeric chemical descriptors
    actual_descriptors = [
        col for col in potential_descriptors_list
        if col in df_metabolites_unique.columns and pd.api.types.is_numeric_dtype(df_metabolites_unique[col])
    ]
    if not actual_descriptors:
        return empty_figure_layout(f"{plot_title} (No numeric chemical descriptors found)")

    # 2. Prepare TF information (significant TFs and their associated metabolites)
    required_tf_cols = ['transcription_factor', 'tf_activity_score', 'tf_activity_pval', 'metabolite_name']
    if not all(col in df_main.columns for col in required_tf_cols):
        return empty_figure_layout(f"{plot_title} (Missing required TF/metabolite columns)")

    tf_metabolite_links = df_main[required_tf_cols].copy()
    tf_metabolite_links.dropna(subset=['transcription_factor', 'metabolite_name', 'tf_activity_score', 'tf_activity_pval'], inplace=True)
    tf_metabolite_links['tf_activity_pval'] = pd.to_numeric(tf_metabolite_links['tf_activity_pval'], errors='coerce')
    tf_metabolite_links.dropna(subset=['tf_activity_pval'], inplace=True)
    significant_tf_links = tf_metabolite_links[tf_metabolite_links['tf_activity_pval'] < p_value_thresh]
    if significant_tf_links.empty:
        return empty_figure_layout(f"{plot_title} (No significant TFs found at p < {p_value_thresh})")

    # 3. Merge with unique metabolite chemical descriptors
    metabolite_chem_profiles = df_metabolites_unique[['metabolite_name'] + actual_descriptors].copy()
    metabolite_chem_profiles.dropna(subset=['metabolite_name'] + actual_descriptors, how='any', inplace=True)
    if metabolite_chem_profiles.empty:
        return empty_figure_layout(f"{plot_title} (No metabolites with complete chemical descriptor data)")

    merged_data = pd.merge(significant_tf_links, metabolite_chem_profiles, on='metabolite_name', how='inner')
    if merged_data.empty:
        return empty_figure_layout(f"{plot_title} (No chemical data for metabolites linked to significant TFs)")
    merged_data.drop_duplicates(subset=['transcription_factor', 'metabolite_name'], inplace=True) # Ensure one profile per TF-Metabolite

    # 4. Determine top active and repressed TFs
    tf_mean_activity = merged_data.groupby('transcription_factor')['tf_activity_score'].mean()
    if tf_mean_activity.empty:
        return empty_figure_layout(f"{plot_title} (Could not calculate mean TF activity scores)")

    top_active_tfs = tf_mean_activity.nlargest(top_n_tfs_each_dir).index.tolist()
    top_repressed_tfs = tf_mean_activity.nsmallest(top_n_tfs_each_dir).index.tolist()
    selected_tfs_list = list(set(top_active_tfs + top_repressed_tfs))
    if not selected_tfs_list:
        return empty_figure_layout(f"{plot_title} (No top TFs identified)")

    # 5. Prepare data for plotting
    plot_df = merged_data[merged_data['transcription_factor'].isin(selected_tfs_list)].copy()
    if plot_df.empty or len(plot_df) < 2: # Parcoords needs at least 2 data points
        return empty_figure_layout(f"{plot_title} (Insufficient data for selected top TFs)")

    # 6. Standardize chemical descriptors
    scaler = StandardScaler()
    plot_df[actual_descriptors] = scaler.fit_transform(plot_df[actual_descriptors])

    # 7. Create Parcoords plot
    unique_tfs_in_plot = sorted(plot_df['transcription_factor'].unique())
    tf_to_int = {tf: i for i, tf in enumerate(unique_tfs_in_plot)}
    plot_df['tf_color_idx'] = plot_df['transcription_factor'].map(tf_to_int)
    
    colorscale_palette = px.colors.qualitative.Plotly 

    dimensions = [dict(label=desc.replace("_", " ").title(), values=plot_df[desc]) for desc in actual_descriptors]

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=plot_df['tf_color_idx'], colorscale=colorscale_palette, showscale=False),
        dimensions=dimensions
    ))

    for i, tf_name in enumerate(unique_tfs_in_plot): # Custom legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(size=10, color=colorscale_palette[i % len(colorscale_palette)]),
                                 name=tf_name, legendgroup=tf_name, showlegend=True))

    fig.update_layout(title_text=plot_title + f"<br>(Significant TF-Metabolite links, p < {p_value_thresh})",
                      height=600, margin=dict(l=80, r=80, t=120, b=80), legend_title_text='Transcription Factor')
    return fig

fig_chem_descriptor_profile_global = generate_chem_descriptor_profile_figure(df_metabolites_unique_global_overview, POTENTIAL_CHEM_DESCRIPTORS)
top_kegg_pathways_names_list_global = df_top_enriched_pathways_global['kegg_pathway_name'].tolist() if not df_top_enriched_pathways_global.empty else []
fig_pathway_descriptor_heatmap_global = generate_pathway_descriptor_heatmap_figure(df, top_kegg_pathways_names_list_global, POTENTIAL_CHEM_DESCRIPTORS)
fig_tf_target_fingerprints_global = generate_tf_target_chem_fingerprints_figure(df, df_metabolites_unique_global_overview, POTENTIAL_CHEM_DESCRIPTORS, P_VALUE_THRESHOLD_GLOBAL, TOP_N_GLOBAL)

# Unique lists for dropdowns in individual explorers
unique_metabolites_dropdown = sorted(df['metabolite_name'].dropna().unique().tolist()) if 'metabolite_name' in df.columns and not df['metabolite_name'].dropna().empty else []
unique_genes_dropdown = sorted(df['gene_symbol'].dropna().unique().tolist()) if 'gene_symbol' in df.columns and not df['gene_symbol'].dropna().empty else []
unique_kegg_pathways_dropdown = sorted(df['kegg_pathway_name'].dropna().unique().tolist()) if 'kegg_pathway_name' in df.columns and not df['kegg_pathway_name'].dropna().empty else []
unique_progeny_pathways_dropdown = sorted(df['progeny_pathway'].dropna().unique().tolist()) if 'progeny_pathway' in df.columns and not df['progeny_pathway'].dropna().empty else []
unique_tfs_dropdown = sorted(df['transcription_factor'].dropna().unique().tolist()) if 'transcription_factor' in df.columns and not df['transcription_factor'].dropna().empty else []


# For Metabolite Explorer
physchem_qm_cols_met_explorer = [
    'qm_total_energy', 'qm_homo', 'qm_lumo', 'qm_gap', 'qm_dipole_moment',
    'xlogp', 'fsp3', 'complexity',
    'hbond_donors', 'hbond_acceptors', 'tpsa',
    'rotatable_bonds', 'mol_weight_da'
]
radar_cols_met_explorer = [
    'mol_weight_da', 'xlogp', 'tpsa', 'hbond_donors', 'hbond_acceptors',
    'rotatable_bonds', 'qm_gap', 'fsp3', 'complexity'
]

# --- Lipinski's Rule of Five Constants & Helper ---
LIPINSKI_MW_MAX = 500
LIPINSKI_LOGP_MAX = 5
LIPINSKI_HDONORS_MAX = 5
LIPINSKI_HACCEPTORS_MAX = 10
# LIPINSKI_ROTATABLE_BONDS_MAX = 10 # Optional, not strictly Ro5

def check_lipinski_rules(metabolite_row):
    rules_status = []
    violations = 0
    if metabolite_row is None or metabolite_row.empty:
        return [{'rule': r, 'value': "N/A", 'passes': 'N/A'} for r in [
            'Molecular Weight (≤ 500 Da)', 'XLogP (≤ 5)', 
            'H-Bond Donors (≤ 5)', 'H-Bond Acceptors (≤ 10)'
        ]], 0

    # Rule 1: Molecular Weight
    mw = metabolite_row.get('mol_weight_da')
    if pd.notna(mw):
        passes = mw <= LIPINSKI_MW_MAX
        rules_status.append({'rule': 'Molecular Weight (≤ 500 Da)', 'value': f"{mw:.2f} Da", 'passes': passes})
        if not passes: violations += 1
    else:
        rules_status.append({'rule': 'Molecular Weight (≤ 500 Da)', 'value': "N/A", 'passes': 'N/A'})

    # Rule 2: LogP
    logp = metabolite_row.get('xlogp')
    if pd.notna(logp):
        passes = logp <= LIPINSKI_LOGP_MAX
        rules_status.append({'rule': 'XLogP (≤ 5)', 'value': f"{logp:.2f}", 'passes': passes})
        if not passes: violations += 1
    else:
        rules_status.append({'rule': 'XLogP (≤ 5)', 'value': "N/A", 'passes': 'N/A'})

    # Rule 3: H-bond Donors
    hbd = metabolite_row.get('hbond_donors')
    if pd.notna(hbd):
        passes = hbd <= LIPINSKI_HDONORS_MAX
        rules_status.append({'rule': 'H-Bond Donors (≤ 5)', 'value': f"{int(hbd) if pd.notna(hbd) else 'N/A'}", 'passes': passes})
        if not passes: violations += 1
    else:
        rules_status.append({'rule': 'H-Bond Donors (≤ 5)', 'value': "N/A", 'passes': 'N/A'})

    # Rule 4: H-bond Acceptors
    hba = metabolite_row.get('hbond_acceptors')
    if pd.notna(hba):
        passes = hba <= LIPINSKI_HACCEPTORS_MAX
        rules_status.append({'rule': 'H-Bond Acceptors (≤ 10)', 'value': f"{int(hba) if pd.notna(hba) else 'N/A'}", 'passes': passes})
        if not passes: violations += 1
    else:
        rules_status.append({'rule': 'H-Bond Acceptors (≤ 10)', 'value': "N/A", 'passes': 'N/A'})
    
    return rules_status, violations

def format_lipinski_rules_html(rules_status, violations):
    if not rules_status or all(item['value'] == "N/A" for item in rules_status):
        return html.P("Lipinski rule data not available for this metabolite.")

    list_items = []
    for item in rules_status:
        status_symbol = "✅" if item['passes'] == True else ("❌" if item['passes'] == False else "❓")
        color = "green" if item['passes'] == True else ("red" if item['passes'] == False else "grey")
        list_items.append(html.Li([f"{item['rule']}: {item['value']} ", html.Span(status_symbol, style={'color': color, 'fontWeight': 'bold'})]))
    
    # Only determine overall pass/fail if we have actual data for violations
    overall_pass_fail_p = html.P("")
    if not all(item['passes'] == 'N/A' for item in rules_status):
        overall_pass = violations <= 1 # Lipinski's rule: no more than one violation
        overall_text = "Passes Lipinski's Ro5 (≤1 violation)" if overall_pass else "Fails Lipinski's Ro5 (>1 violation)"
        overall_color = "green" if overall_pass else "red"
        overall_pass_fail_p = html.P(html.Strong(f"Overall: {overall_text} ({violations} violations)"), style={'color': overall_color, 'marginTop': '10px'})

    return html.Div([html.H5("Lipinski's Rule of Five Status:", style={'marginBottom': '10px'}), html.Ul(list_items, style={'listStyleType': 'none', 'paddingLeft': '0'}), overall_pass_fail_p])


# For Integrative Viewer (Network and Correlation)
def process_dataframe_for_graph_and_dropdowns(dataframe):
    all_nodes_cytoscape = []
    all_edges_cytoscape = []
    typed_node_dropdown_options = {'metabolite': [], 'gene': [], 'pathway': [], 'tf': []}
    node_details_map = {}
    temp_nodes_dict = {}
    unique_edges_set = set()
    node_to_edges_map = {} # New structure for efficient lookup

    if dataframe.empty:
        # Return empty structures, including the new map
        return all_nodes_cytoscape, all_edges_cytoscape, typed_node_dropdown_options, node_details_map, node_to_edges_map
        
    for _, row in dataframe.iterrows():
        entities = {
            'metabolite': row.get('metabolite_name'),
            'gene': row.get('gene_symbol'),
            'pathway': row.get('kegg_pathway_name'),
            'tf': row.get('transcription_factor')
        }
        entity_ids = {}
        for entity_type, entity_name in entities.items():
            if pd.notna(entity_name):
                node_id = f"{entity_type[:4]}_{str(entity_name)}" # Ensure entity_name is string
                entity_ids[entity_type] = node_id
                if node_id not in temp_nodes_dict:
                    temp_nodes_dict[node_id] = {'id': node_id, 'label': str(entity_name), 'type': entity_type, 'full_data': { 'name': str(entity_name), 'original_column': f'{entity_type}_name' if entity_type != 'tf' else 'transcription_factor'}}
        
        met_id, gene_id, path_id, tf_id = entity_ids.get('metabolite'), entity_ids.get('gene'), entity_ids.get('pathway'), entity_ids.get('tf')
        if met_id and gene_id: unique_edges_set.add(tuple(sorted((met_id, gene_id))) + ('met_gene_assoc',))
        if gene_id and path_id: unique_edges_set.add(tuple(sorted((gene_id, path_id))) + ('gene_in_pathway',))
        if tf_id and gene_id: unique_edges_set.add((tf_id, gene_id, 'tf_regulates_gene'))
        if met_id and path_id: unique_edges_set.add(tuple(sorted((met_id, path_id))) + ('met_in_pathway',))

    for node_id, attrs in temp_nodes_dict.items():
        all_nodes_cytoscape.append({'data': attrs})
        node_details_map[node_id] = attrs
        if attrs['type'] in typed_node_dropdown_options:
            typed_node_dropdown_options[attrs['type']].append({'label': attrs['label'], 'value': node_id})
            
    for source, target, relationship in unique_edges_set:
        edge_element = {'data': {'source': source, 'target': target, 'relationship': relationship}}
        all_edges_cytoscape.append(edge_element)
        
        # Populate the node_to_edges_map
        if source not in node_to_edges_map: node_to_edges_map[source] = []
        if target not in node_to_edges_map: node_to_edges_map[target] = []
        node_to_edges_map[source].append(edge_element)
        # Add the edge to the target's list as well for non-directional edges
        # Assuming 'tf_regulates_gene' is the only directional one
        if relationship != 'tf_regulates_gene':
             node_to_edges_map[target].append(edge_element)
        
    for entity_type in typed_node_dropdown_options:
        typed_node_dropdown_options[entity_type] = sorted(typed_node_dropdown_options[entity_type], key=lambda x: x['label'])
    return all_nodes_cytoscape, all_edges_cytoscape, typed_node_dropdown_options, node_details_map, node_to_edges_map

ALL_NODES_CYTOSCAPE, ALL_EDGES_CYTOSCAPE, TYPED_NODE_DROPDOWN_OPTIONS, NODE_DETAILS_MAP, NODE_TO_EDGES_MAP = process_dataframe_for_graph_and_dropdowns(df)

NUMERICAL_COLS_CORR = list(df.select_dtypes(include=np.number).columns) if not df.empty else []
CATEGORICAL_COLS_CORR_COLOR = ['None']
if not df.empty:
    potential_color_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    bool_cols = list(df.select_dtypes(include=['bool']).columns) # Booleans can also be used for color
    potential_color_cols.extend(bool_cols)

    for col in potential_color_cols:
        if col in df.columns and 1 < df[col].nunique() < 30: # Only add if has a few unique values
            CATEGORICAL_COLS_CORR_COLOR.append(col)
    common_cats = ['direction_flag', 'class_i_flag', 'class_ii_flag', 'progeny_pathway', 'significant_metabolite', 'significant_gene']
    for cc in common_cats:
        if cc in df.columns and cc not in CATEGORICAL_COLS_CORR_COLOR and 1 < df[cc].nunique() < 30:
            CATEGORICAL_COLS_CORR_COLOR.append(cc)
CATEGORICAL_COLS_CORR_COLOR = sorted(list(set(CATEGORICAL_COLS_CORR_COLOR)))
if 'None' not in CATEGORICAL_COLS_CORR_COLOR: CATEGORICAL_COLS_CORR_COLOR.insert(0,'None')


cytoscape_stylesheet = [
    {'selector': 'node', 'style': {'label': 'data(label)', 'width': '70px', 'height': '70px', 'font-size': '10px', 'text-valign': 'bottom', 'text-halign': 'center','background-color': '#FFFFFF', 'border-width': '2px', 'border-color': 'black', 'text-wrap': 'wrap', 'text-max-width': '60px'}},
    {'selector': '[type = "metabolite"]', 'style': {'background-color': '#FF6347', 'shape': 'ellipse'}},
    {'selector': '[type = "gene"]', 'style': {'background-color': '#4682B4', 'shape': 'rectangle'}},
    {'selector': '[type = "pathway"]', 'style': {'background-color': '#32CD32', 'shape': 'round-hexagon'}},
    {'selector': '[type = "tf"]', 'style': {'background-color': '#FFD700', 'shape': 'diamond'}},
    {'selector': '[type = "placeholder"]', 'style': {'background-color': '#808080', 'shape': 'star'}},
    {'selector': 'edge', 'style': {'width': 2, 'line-color': '#cccccc', 'target-arrow-color': '#cccccc','target-arrow-shape': 'triangle', 'curve-style': 'bezier'}},
    {'selector': '[relationship = "tf_regulates_gene"]', 'style': {'line-color': '#FFD700', 'target-arrow-color': '#FFD700'}},
    {'selector': ':selected', 'style': {'border-width': 4, 'border-color': '#0074D9', 'line-color': '#0074D9','target-arrow-color': '#0074D9'}}
]

# --- Helper Functions for Layouts ---
def create_summary_card(title, value, card_id):
    return html.Div([
        html.H4(title, className="card-title"),
        html.P(str(value), className="card-text", style={'fontSize': '24px', 'fontWeight': 'bold'})
    ], className="card", id=card_id, style={'textAlign': 'center', 'padding': '10px', 'margin': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'flex': '1', 'minWidth': '200px'})

# --- Helper Function for Property Bar Charts ---
def create_property_bar_chart(data_row, properties_config, chart_title, yaxis_title="Value"):
    labels = []
    values = []
    valid_data_found = False

    if data_row is None or data_row.empty:
        pass # Will fall through to no data message
    else:
        for prop_info in properties_config:
            prop_id = prop_info['id']
            prop_label = prop_info['label']
            value = data_row.get(prop_id)

            if pd.notna(value) and isinstance(value, (int, float)):
                labels.append(prop_label)
                values.append(value)
                valid_data_found = True
            # else: # Optionally, show missing properties with a placeholder
            #     labels.append(prop_label)
            #     values.append(0) # Or np.nan - Plotly won't plot np.nan bars

    if not valid_data_found:
        fig = go.Figure()
        fig.update_layout(title_text=f"{chart_title} (No data)", annotations=[dict(text="No data to display", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)], height=300, margin=dict(t=50, b=20, l=20, r=20))
        return fig

    fig = go.Figure(data=[go.Bar(x=labels, y=values, text=[f'{v:.2f}' if isinstance(v, float) else str(v) for v in values], textposition='auto', marker_color='skyblue')])
    fig.update_layout(
        title_text=chart_title,
        xaxis_title="Property",
        yaxis_title=yaxis_title,
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis={'categoryorder':'array', 'categoryarray': labels} # Keep original order
    )
    return fig



# --- Tab Layout Creation Functions ---

def create_global_overview_tab():
    return html.Div([
        html.H2("Global Data Overview", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            create_summary_card("Unique Metabolites", num_unique_metabolites_global, "card-metabolites-global"),
            create_summary_card("Unique Genes", num_unique_genes_global, "card-genes-global"),
            create_summary_card("Unique KEGG Pathways", num_unique_kegg_pathways_global, "card-pathways-global"),
            create_summary_card("Unique TFs", num_unique_tfs_global, "card-tfs-global"),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
        html.Div([
            create_summary_card(f"Significant Metabolites (p < {P_VALUE_THRESHOLD_GLOBAL})", significant_metabolites_count_global, "card-sig-met-global"),
            create_summary_card(f"Significant Genes (p < {P_VALUE_THRESHOLD_GLOBAL})", significant_genes_count_global, "card-sig-gene-global"),
            create_summary_card(f"Significant TFs (p < {P_VALUE_THRESHOLD_GLOBAL})", significant_tfs_count_global, "card-sig-tf-global"),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '40px'}),
        
        html.Div([ # Distributions Row
            html.Div([
                dcc.Graph(figure=fig_met_dist_global),
                html.Div([html.H5("💡 Metabolite Log2FC", style=sticky_note_title_style), html.P("Distribution of log2 fold changes for unique metabolites. Shows overall up/down-regulation trends.")], style=sticky_note_style)
            ], style={'flex': '1 1 300px', 'minWidth': '300px', 'padding': '10px', 'boxSizing': 'border-box'}),
            html.Div([
                dcc.Graph(figure=fig_gene_dist_global),
                html.Div([html.H5("💡 Gene Log2FC", style=sticky_note_title_style), html.P("Distribution of log2 fold changes for unique genes.")], style=sticky_note_style)
            ], style={'flex': '1 1 300px', 'minWidth': '300px', 'padding': '10px', 'boxSizing': 'border-box'}),
            html.Div([
                dcc.Graph(figure=fig_tf_dist_global),
                html.Div([html.H5("💡 TF Activity Scores", style=sticky_note_title_style), html.P("Distribution of activity scores for unique transcription factors.")], style=sticky_note_style)
            ], style={'flex': '1 1 300px', 'minWidth': '300px', 'padding': '10px', 'boxSizing': 'border-box'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '40px'}),
        
        html.Div([ # Pie Chart Row
            dcc.Graph(figure=fig_direction_pie_global),
            html.Div([html.H5("💡 Metabolite Regulation", style=sticky_note_title_style), html.P("Proportion of unique metabolites that are up-regulated versus down-regulated based on their 'direction_flag'.")], style=sticky_note_style)
        ], style={'maxWidth': '600px', 'width': '90%', 'margin': 'auto', 'marginBottom': '40px'}),
        
        html.H3(f"Top Hits (p < {P_VALUE_THRESHOLD_GLOBAL})", style={'textAlign': 'center', 'marginTop': '40px'}),
        html.Div([ # Top Hits Row
            html.Div([dcc.Graph(figure=fig_top_met_global)], style={'width': '98%', 'margin': 'auto', 'marginBottom': '5px'}),
            html.Div([html.H5("💡 Top Metabolites", style=sticky_note_title_style), html.P(f"Shows the top {TOP_N_GLOBAL} up- and down-regulated significant metabolites.")], style={**sticky_note_style, 'width': '98%', 'margin': 'auto', 'marginBottom': '20px'}),
            
            html.Div([dcc.Graph(figure=fig_top_gene_global)], style={'width': '98%', 'margin': 'auto', 'marginBottom': '5px'}),
            html.Div([html.H5("💡 Top Genes", style=sticky_note_title_style), html.P(f"Shows the top {TOP_N_GLOBAL} up- and down-regulated significant genes.")], style={**sticky_note_style, 'width': '98%', 'margin': 'auto', 'marginBottom': '20px'}),

            html.Div([dcc.Graph(figure=fig_top_tf_global)], style={'width': '98%', 'margin': 'auto', 'marginBottom': '5px'}),
            html.Div([html.H5("💡 Top TFs", style=sticky_note_title_style), html.P(f"Shows the top {TOP_N_GLOBAL} most active and inactive significant TFs by activity score.")], style={**sticky_note_style, 'width': '98%', 'margin': 'auto', 'marginBottom': '20px'}),

            html.Div([dcc.Graph(figure=fig_top_enriched_pathways_global)], style={'width': '98%', 'margin': 'auto', 'marginBottom': '5px'}),
            html.Div([html.H5("💡 Top Enriched Pathways", style=sticky_note_title_style), html.P(f"Shows the top {TOP_N_PATHWAYS_GLOBAL} KEGG pathways ranked by their pathway score. Higher scores indicate greater enrichment or activity.")], style={**sticky_note_style, 'width': '98%', 'margin': 'auto', 'marginBottom': '40px'}),
        ]),
        
        html.H3("Volcano Plots", style={'textAlign': 'center', 'marginTop': '40px'}),
        html.Div([ # Volcano Plots Row
            html.Div([
                dcc.Graph(figure=fig_volcano_met_global),
                html.Div([html.H5("💡 Metabolite Volcano Plot", style=sticky_note_title_style), html.P("Visualizes statistical significance (Y-axis) against magnitude of change (X-axis) for metabolites. Red points are significant based on set thresholds (dashed lines).")], style=sticky_note_style)
            ], style={'flex': '1 1 400px', 'minWidth': '350px', 'padding': '10px', 'boxSizing': 'border-box'}),
            html.Div([
                dcc.Graph(figure=fig_volcano_gene_global),
                html.Div([html.H5("💡 Gene Volcano Plot", style=sticky_note_title_style), html.P("Similar to the metabolite volcano plot, but for gene expression data. Red points highlight significantly altered genes.")], style=sticky_note_style)
            ], style={'flex': '1 1 400px', 'minWidth': '350px', 'padding': '10px', 'boxSizing': 'border-box'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '40px'}),

        html.H3("Chemical Descriptor Profiles", style={'textAlign': 'center', 'marginTop': '40px'}),
        html.Div([ # Chemical Descriptor Profile Plot Row
            html.Div([
                dcc.Graph(id='chem-descriptor-profile-plot-global', figure=fig_chem_descriptor_profile_global),
                html.Div([
                    html.H5("💡 Chemical Descriptor Profiles by Regulation", style=sticky_note_title_style),
                    html.P("This plot compares standardized chemical descriptor profiles (Z-scores) between up-regulated and down-regulated metabolites. Descriptors are ordered by their importance in distinguishing these groups, determined by a Random Forest model with cross-validation (if data permits, otherwise alphabetical). Lines show mean profiles ± 95% CI. Individual metabolite traces are shown faintly. '*' indicates significant difference (Mann-Whitney U, p < 0.05). Yellow callouts highlight the top 3 descriptors with the largest absolute mean difference between groups.")
                ], style=sticky_note_style)
            ], style={'width': '98%', 'margin': 'auto', 'marginBottom': '20px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
        
        html.H3("Pathway Chemical Signatures", style={'textAlign': 'center', 'marginTop': '40px'}),
        html.Div([ # Pathway-Descriptor Heatmap Row
            html.Div([
                dcc.Graph(id='pathway-descriptor-heatmap-global', figure=fig_pathway_descriptor_heatmap_global),
                html.Div([
                    html.H5("💡 Pathway-Level Mean Chemical Descriptor Z-scores", style=sticky_note_title_style),
                    html.P("This heatmap shows the average Z-scores of chemical descriptors for metabolites within the top enriched KEGG pathways. Each cell represents the mean standardized value of a descriptor (column) for a given pathway (row, ordered by enrichment score). Red indicates higher-than-average Z-scores for that descriptor within the pathway's metabolites, blue indicates lower-than-average, and white is near zero. This helps identify if certain chemical properties are characteristic of metabolites in highly active/enriched pathways.")
                ], style=sticky_note_style)
            ], style={'width': '98%', 'margin': 'auto', 'marginBottom': '20px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '20px'}),

        html.H3("TF Target Chemical Fingerprints", style={'textAlign': 'center', 'marginTop': '40px'}),
        html.Div([ # TF Target Chemical Fingerprints Parallel Coordinates Plot Row
            html.Div([
                dcc.Graph(id='tf-target-fingerprints-plot-global', figure=fig_tf_target_fingerprints_global),
                html.Div([
                    html.H5("💡 Chemical Profiles of Metabolites Linked to Top TFs", style=sticky_note_title_style),
                    html.P(f"This parallel coordinates plot displays the standardized chemical descriptor profiles (Z-scores) of metabolites associated with the top {TOP_N_GLOBAL} most active and top {TOP_N_GLOBAL} most repressed TFs (based on significant TF-metabolite links with p < {P_VALUE_THRESHOLD_GLOBAL}). Each line represents a metabolite, colored by its associated TF. This helps visualize if metabolites linked to certain TFs share common chemical characteristics.")
                ], style=sticky_note_style)
            ], style={'width': '98%', 'margin': 'auto', 'marginBottom': '20px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
        
    ], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})

def create_metabolite_explorer_tab():
    return html.Div([
        html.H2("Metabolite Explorer", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='metabolite-dropdown-explorer',
            options=[{'label': met, 'value': met} for met in unique_metabolites_dropdown],
            value=unique_metabolites_dropdown[0] if unique_metabolites_dropdown else None,
            clearable=False,
            style={'width': '50%', 'margin': '0 auto 20px auto'}
        ),
        html.Div(id='metabolite-explorer-explanation', style=sticky_note_style),
        html.Div(id='metabolite-details-output-explorer', children=[ # Parent Div
            # Define static Divs for cards here
            html.Div([ # Row for cards
                html.Div(id='metabolite-id-reg-card-explorer', className='details-card', style={'width': '45%', 'padding': '10px', 'border': '1px solid #ddd', 'margin': '5px', 'minHeight':'150px'}),
                html.Div(id='metabolite-classification-card-explorer', className='details-card', style={'width': '45%', 'padding': '10px', 'border': '1px solid #ddd', 'margin': '5px', 'minHeight':'150px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}),
            html.H3("Physicochemical & QM Properties", style={'textAlign': 'center', 'marginTop': '20px'}),
            html.Div([html.H5("💡 Chemical Property Bar Charts", style=sticky_note_title_style), html.P("These bar charts display groups of key chemical properties with their actual (unnormalized) values, allowing for direct comparison. Hover over bars for exact values.")], style=sticky_note_style),
            html.Div([ # Row for Bar Charts
                html.Div(dcc.Graph(id='metabolite-bar-chart-size-explorer'), style={'flex': '1 1 300px', 'minWidth': '300px', 'padding': '5px'}),
                html.Div(dcc.Graph(id='metabolite-bar-chart-polarity-explorer'), style={'flex': '1 1 300px', 'minWidth': '300px', 'padding': '5px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '10px'}),
            html.Div([ # Another Row for Bar Charts
                html.Div(dcc.Graph(id='metabolite-bar-chart-lipophilicity-explorer'), style={'flex': '1 1 300px', 'minWidth': '300px', 'padding': '5px'}),
                html.Div(dcc.Graph(id='metabolite-bar-chart-qm-explorer'), style={'flex': '1 1 300px', 'minWidth': '300px', 'padding': '5px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
            
            html.H3("🔍 Chemical Disposition Heuristics", style={'textAlign': 'center', 'marginTop': '20px'}),
            html.Div([
                html.H5("💡 Chemical Disposition Heuristics (adapted from Lipinski’s Rules)", style=sticky_note_title_style), 
                html.P(
                    "These rules, while originally designed for drug discovery, offer insight into a molecule’s "
                    "permeability, polarity, and cellular accessibility — relevant to understanding metabolite "
                    "regulation in tumor biology."
                )], style=sticky_note_style),
            html.Div(id='metabolite-lipinski-rules-explorer', style={'width': '60%', 'margin': '10px auto', 'padding': '15px', 'border': '1px solid #eee', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'}),

            html.H4("Detailed Properties Table & Radar Plot", style={'textAlign': 'center', 'marginTop': '30px'}),
            html.Div(id='metabolite-physchem-qm-table-explorer', style={'margin': '10px auto', 'width': '80%'}),
            html.Div(dcc.Graph(id='metabolite-radar-plot-explorer'), style={'width': '70%', 'minWidth':'400px', 'margin': '10px auto'}),
            html.H3("Associated Genes", style={'textAlign': 'center', 'marginTop': '30px'}),
            html.Div(id='associated-genes-table-explorer', style={'margin': '10px auto', 'width': '80%'}),
            html.H3("Associated KEGG Information", style={'textAlign': 'center', 'marginTop': '30px'}),
            html.Div(id='associated-kegg-table-explorer', style={'margin': '10px auto', 'width': '80%'}),
            html.H3("Associated Transcription Factors", style={'textAlign': 'center', 'marginTop': '30px'}),
            html.Div(id='associated-tf-table-explorer', style={'margin': '10px auto', 'width': '80%'}),
        ])
    ], style={'padding': '20px'})

def create_gene_explorer_tab():
    return html.Div([
        html.H2("Gene Explorer", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='gene-dropdown-explorer',
            options=[{'label': gene, 'value': gene} for gene in unique_genes_dropdown],
            value=unique_genes_dropdown[0] if unique_genes_dropdown else None,
            clearable=False,
            style={'width': '50%', 'margin': '0 auto 20px auto'}
        ),
        html.Div(id='gene-explorer-explanation', style=sticky_note_style),
        html.Div(id='gene-details-output-explorer', children=[ # Parent Div
            # Define static Div for card here
            html.Div(id='gene-id-expression-card-explorer', className='details-card', style={'width': 'fit-content', 'maxWidth': '90%', 'padding': '15px', 'border': '1px solid #ddd', 'margin': '10px auto', 'textAlign': 'center', 'minHeight':'100px'}),
            html.H3("Associated Metabolites", style={'textAlign': 'center', 'marginTop': '30px'}),
            html.Div(id='associated-metabolites-table-gene-explorer', style={'margin': '10px auto', 'width': '80%'}),
            html.H3("Associated KEGG Information", style={'textAlign': 'center', 'marginTop': '30px'}),
            html.Div(id='associated-kegg-info-table-gene-explorer', style={'margin': '10px auto', 'width': '80%'}),
            html.H3("Associated Transcription Factors", style={'textAlign': 'center', 'marginTop': '30px'}),
            html.Div(id='associated-tf-info-table-gene-explorer', style={'margin': '10px auto', 'width': '80%'}),
        ])
    ], style={'padding': '20px'})

def create_pathway_explorer_tab():
    return html.Div([
        html.H2("Pathway Explorer", style={'textAlign': 'center'}),
        dcc.RadioItems(
            id='pathway-type-selector-explorer',
            options=[
                {'label': 'KEGG Pathways', 'value': 'KEGG'},
                {'label': 'PROGENy Pathways', 'value': 'PROGENy'},
            ],
            value='KEGG',
            labelStyle={'display': 'inline-block', 'marginRight': '20px'},
            style={'textAlign': 'center', 'margin': '20px'}
        ),
        dcc.Dropdown(
            id='pathway-dropdown-explorer',
            clearable=False,
            style={'width': '70%', 'margin': '0 auto 20px auto'}
        ),
        html.Div(id='pathway-explorer-explanation', style=sticky_note_style),
        html.Div(id='pathway-details-output-explorer', style={'padding': '20px'}) # Content generated by callback
    ], style={'padding': '20px'})

def create_tf_explorer_tab():
    return html.Div([
        html.H2("Transcription Factor (TF) Explorer", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='tf-dropdown-explorer',
            options=[{'label': tf, 'value': tf} for tf in unique_tfs_dropdown],
            value=unique_tfs_dropdown[0] if unique_tfs_dropdown else None,
            clearable=False,
            style={'width': '50%', 'margin': '0 auto 20px auto'}
        ),
        html.Div(id='tf-explorer-explanation', style=sticky_note_style),
        html.Div(id='tf-details-output-explorer', children=[ # Parent Div
            # Define static Div for card here
            html.Div(id='tf-id-activity-card-explorer', className='details-card', style={'width': 'fit-content', 'maxWidth': '90%', 'padding': '15px', 'border': '1px solid #ddd', 'margin': '10px auto', 'textAlign': 'center', 'minHeight':'80px'}),
            html.Div(dcc.Graph(id='tf-differential-activity-plot-explorer'), style={'width': '70%', 'minWidth':'400px', 'margin': '20px auto'}),
            html.H3("Associated Genes (Potential Targets)", style={'textAlign': 'center', 'marginTop': '30px'}),
            html.Div(id='associated-genes-table-tf-explorer', style={'margin': '10px auto', 'width': '80%'}),
            html.H3("Associated Metabolites", style={'textAlign': 'center', 'marginTop': '30px'}),
            html.Div(id='associated-metabolites-table-tf-explorer', style={'margin': '10px auto', 'width': '80%'}),
            html.H3("Associated KEGG Pathways", style={'textAlign': 'center', 'marginTop': '30px'}),
            html.Div(id='associated-kegg-pathways-table-tf-explorer', style={'margin': '10px auto', 'width': '80%'}),
        ])
    ], style={'padding': '20px'})

def create_network_explorer_tab():
    return html.Div([
        html.H2("Explore Neighborhood Networks", style={'textAlign': 'center', 'marginTop': '20px'}),
        html.Div(id='network-explorer-explanation', style=sticky_note_style),
        html.Div([
            html.Label("Select Node Type:", style={'marginRight': '10px'}),
            dcc.RadioItems(
                id='network-node-type-selector-explorer',
                options=[
                    {'label': 'Metabolite', 'value': 'metabolite'},
                    {'label': 'Gene', 'value': 'gene'},
                    {'label': 'Pathway', 'value': 'pathway'},
                    {'label': 'TF', 'value': 'tf'}
                ],
                value='metabolite',
                labelStyle={'display': 'inline-block', 'marginRight': '15px'},
                style={'marginBottom': '10px'}
            ),
        ], style={'textAlign': 'center'}),
        html.Div([
            html.Label("Select Central Node:", style={'marginRight': '10px'}),
            dcc.Dropdown(id='network-central-node-dropdown-explorer', style={'width': '90%', 'maxWidth': '500px', 'minWidth':'250px', 'display': 'inline-block', 'margin': 'auto'}),
        ], style={'textAlign': 'center', 'margin': '10px 0 20px 0'}),
        cyto.Cytoscape(
            id='cytoscape-neighborhood-network-explorer',
            elements=[],
            style={'width': '100%', 'height': '550px', 'border': '1px solid #ccc'},
            layout={'name': 'cose', 'idealEdgeLength': 100, 'nodeOverlap': 20, 'padding': 30, 'animate': False, 'fit': True}, # Animate can be True or False
            stylesheet=cytoscape_stylesheet
        ),
        html.Pre(id='tap-node-data-output-explorer', style={'border': '1px solid lightgrey', 'padding': '10px', 'marginTop': '10px', 'height': '150px', 'overflowY': 'auto', 'fontSize': '12px', 'backgroundColor': '#f9f9f9', 'whiteSpace': 'pre-wrap'})
    ], style={'padding': '20px'})

# --- Main App Layout ---
app.layout = html.Div([
    html.H1("Comprehensive Multi-Omics Dashboard", style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f0f0f0'}),
    dcc.Tabs(id="main-tabs", value='tab-global-overview', children=[
        dcc.Tab(label='Global Overview', value='tab-global-overview', children=[create_global_overview_tab()]),
        dcc.Tab(label='Metabolite Explorer', value='tab-metabolite-explorer', children=[create_metabolite_explorer_tab()]),
        dcc.Tab(label='Gene Explorer', value='tab-gene-explorer', children=[create_gene_explorer_tab()]),
        dcc.Tab(label='Pathway Explorer', value='tab-pathway-explorer', children=[create_pathway_explorer_tab()]),
        dcc.Tab(label='TF Explorer', value='tab-tf-explorer', children=[create_tf_explorer_tab()]),
        dcc.Tab(label='Network Explorer', value='tab-network-explorer', children=[create_network_explorer_tab()]),
    ])
])

# --- Callbacks ---

# Metabolite Explorer Callbacks
@app.callback(
    [Output('metabolite-id-reg-card-explorer', 'children'),
     Output('metabolite-classification-card-explorer', 'children'),
     Output('metabolite-physchem-qm-table-explorer', 'children'),
     Output('metabolite-radar-plot-explorer', 'figure'),
     Output('metabolite-bar-chart-size-explorer', 'figure'),
     Output('metabolite-bar-chart-polarity-explorer', 'figure'),
     Output('metabolite-bar-chart-lipophilicity-explorer', 'figure'),
     Output('metabolite-bar-chart-qm-explorer', 'figure'),
     Output('metabolite-lipinski-rules-explorer', 'children'),
     Output('associated-genes-table-explorer', 'children'),
     Output('associated-kegg-table-explorer', 'children'),
     Output('associated-tf-table-explorer', 'children'),
     Output('metabolite-explorer-explanation', 'children')],
    [Input('metabolite-dropdown-explorer', 'value')]
)
def update_metabolite_details_explorer(selected_metabolite):
    # Property groups for bar charts
    PROP_GROUP_SIZE_COMPLEXITY_MET = [{'id': 'mol_weight_da', 'label': 'Mol. Weight (Da)'}, {'id': 'complexity', 'label': 'Complexity'}, {'id': 'fsp3', 'label': 'FSP3'}]
    PROP_GROUP_POLARITY_HBOND_MET = [{'id': 'tpsa', 'label': 'TPSA (Å²)'}, {'id': 'hbond_donors', 'label': 'H-Bond Donors'}, {'id': 'hbond_acceptors', 'label': 'H-Bond Acceptors'}, {'id': 'qm_dipole_moment', 'label': 'Dipole Moment (Debye)'}]
    PROP_GROUP_LIPOPHILICITY_MET = [{'id': 'xlogp', 'label': 'XLogP'}]
    PROP_GROUP_QM_ENERGIES_MET = [{'id': 'qm_homo', 'label': 'HOMO (eV)'}, {'id': 'qm_lumo', 'label': 'LUMO (eV)'}, {'id': 'qm_gap', 'label': 'HOMO-LUMO Gap (eV)'}]

    no_data_card_msg = html.P("N/A", style={'textAlign': 'center', 'paddingTop':'20px'})
    no_data_table_msg = html.P("No data available.", style={'textAlign': 'center'})
    empty_fig = go.Figure(layout={'annotations': [{'text': 'Select a metabolite to view its radar plot.', 'showarrow': False, 'font': {'size': 14}}]})
    empty_bar_fig = go.Figure(layout={'height':300, 'annotations': [{'text': 'Select metabolite.', 'showarrow': False}]})
    default_lipinski_html = html.P("Select a metabolite to see Lipinski's Rules.", style={'textAlign': 'center'})
    
    default_explanation = html.Div([
        html.H5("💡 Metabolite Explorer", style=sticky_note_title_style), 
        html.P("Use the dropdown to select a metabolite. Its properties and associations will be displayed below. Bar charts show key chemical properties (Size/Complexity, Polarity/H-Bonding, Lipophilicity, QM Energies) with their actual values. Lipinski's Rule of Five indicators assess drug-likeness.")
    ]) # This default explanation might also need an update to reflect the new section title if it refers to it.

    if not selected_metabolite or df.empty or 'metabolite_name' not in df.columns:
        return (no_data_card_msg, no_data_card_msg, no_data_table_msg, empty_fig,
                empty_bar_fig, empty_bar_fig, empty_bar_fig, empty_bar_fig, default_lipinski_html,
                no_data_table_msg, no_data_table_msg, no_data_table_msg, default_explanation)

    metabolite_data_unique_row_df = df[df['metabolite_name'] == selected_metabolite].drop_duplicates(subset=['metabolite_name'])
    if metabolite_data_unique_row_df.empty:
        no_detail_msg = html.P(f"No detailed data found for {selected_metabolite}.")
        return (no_data_card_msg, no_data_card_msg, no_detail_msg, empty_fig,
                empty_bar_fig, empty_bar_fig, empty_bar_fig, empty_bar_fig, default_lipinski_html,
                no_data_table_msg, no_data_table_msg, no_data_table_msg, default_explanation)
    
    metabolite_data_unique_row = metabolite_data_unique_row_df.iloc[0]
    metabolite_associated_rows = df[df['metabolite_name'] == selected_metabolite]

    id_reg_content_children = [
        html.H4("Identification & Regulation"),
        html.P(f"Name: {metabolite_data_unique_row.get('metabolite_name', 'N/A')}"),
        html.P(f"Direction: {metabolite_data_unique_row.get('direction_flag', 'N/A')}"),
        html.P(f"Log2FC: {metabolite_data_unique_row.get('metabolite_log2fc', float('nan')):.2f}" if pd.notna(metabolite_data_unique_row.get('metabolite_log2fc')) else "Log2FC: N/A"),
        html.P(f"Fold Change: {metabolite_data_unique_row.get('metabolite_fold_change', float('nan')):.2f}" if pd.notna(metabolite_data_unique_row.get('metabolite_fold_change')) else "Fold Change: N/A"),
        html.P(f"P-value: {metabolite_data_unique_row.get('metabolite_pval', float('nan')):.2e}" if pd.notna(metabolite_data_unique_row.get('metabolite_pval')) else "P-value: N/A"),
        html.P(f"VIP Score: {metabolite_data_unique_row.get('vip_score', float('nan')):.2f}" if pd.notna(metabolite_data_unique_row.get('vip_score')) else "VIP Score: N/A"),
    ]
    classification_content_children = [
        html.H4("Chemical Classification"),
        html.P(f"Class I: {metabolite_data_unique_row.get('class_i_flag', 'N/A')}"),
        html.P(f"Class II: {metabolite_data_unique_row.get('class_ii_flag', 'N/A')}"),
    ]

    valid_physchem_cols = [col for col in physchem_qm_cols_met_explorer if col in metabolite_data_unique_row and pd.notna(metabolite_data_unique_row[col])]
    if valid_physchem_cols:
        physchem_qm_data = metabolite_data_unique_row[valid_physchem_cols].copy().astype(str) # Convert to string for display
        # Smart formatting can be added here if needed
        physchem_table_df = pd.DataFrame(physchem_qm_data).reset_index()
        physchem_table_df.columns = ['Property', 'Value']
        physchem_table_display = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in physchem_table_df.columns],
            data=physchem_table_df.to_dict('records'),
            style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal', 'height': 'auto', 'minWidth': '100px', 'maxWidth': '200px'}, style_header={'fontWeight': 'bold'}, style_as_list_view=True,
        )
    else:
        physchem_table_display = html.P("No physicochemical properties available.", style={'textAlign':'center'})


    radar_values = []
    valid_radar_cols_for_plot = [] # Use a different name to avoid conflict
    for col in radar_cols_met_explorer:
        value = metabolite_data_unique_row.get(col)
        if pd.notna(value) and isinstance(value, (int, float)):
            radar_values.append(value)
            valid_radar_cols_for_plot.append(col)
        else: # Add 0 for missing or non-numeric, so radar plot has all categories
            radar_values.append(0) 
            valid_radar_cols_for_plot.append(col)

    radar_fig_updated = go.Figure()
    if valid_radar_cols_for_plot and any(v != 0 for v in radar_values): # Only plot if there's some data
        radar_fig_updated.add_trace(go.Scatterpolar(r=radar_values, theta=valid_radar_cols_for_plot, fill='toself', name=selected_metabolite))
        max_val = max(radar_values) if radar_values else 1
        min_val = min(0, min(radar_values) if radar_values else 0)
        radar_fig_updated.update_layout(polar=dict(radialaxis=dict(visible=True, range=[min_val, max_val])), title=f"Profile for {selected_metabolite}", height=400)
    else:
        radar_fig_updated.update_layout(annotations=[{'text': 'Radar plot data not sufficient or unavailable.', 'showarrow': False, 'font': {'size': 14}}], height=400)
    
    # Create bar charts
    bar_chart_size = create_property_bar_chart(metabolite_data_unique_row, PROP_GROUP_SIZE_COMPLEXITY_MET, "Size & Complexity")
    bar_chart_polarity = create_property_bar_chart(metabolite_data_unique_row, PROP_GROUP_POLARITY_HBOND_MET, "Polarity & H-Bonding")
    bar_chart_lipophilicity = create_property_bar_chart(metabolite_data_unique_row, PROP_GROUP_LIPOPHILICITY_MET, "Lipophilicity (XLogP)")
    bar_chart_qm = create_property_bar_chart(metabolite_data_unique_row, PROP_GROUP_QM_ENERGIES_MET, "QM Energies")
    lipinski_status, lipinski_violations = check_lipinski_rules(metabolite_data_unique_row)
    lipinski_html = format_lipinski_rules_html(lipinski_status, lipinski_violations)

    genes_df_cols = ['gene_symbol', 'gene_log2fc', 'gene_expr_pval']
    if all(c in metabolite_associated_rows.columns for c in genes_df_cols):
        genes_df = metabolite_associated_rows[genes_df_cols].drop_duplicates().copy()
        if not genes_df.empty:
            genes_df['gene_log2fc'] = genes_df['gene_log2fc'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            genes_df['gene_expr_pval'] = genes_df['gene_expr_pval'].map(lambda x: f"{x:.2e}" if pd.notna(x) else "N/A")
        genes_table = dash_table.DataTable(columns=[{"name": i.replace("_", " ").title(), "id": i} for i in genes_df.columns], data=genes_df.to_dict('records'), page_size=5, style_cell={'textAlign': 'left', 'padding': '5px'}, style_header={'fontWeight': 'bold'}) if not genes_df.empty else html.P(f"No genes associated with {selected_metabolite}.", style={'textAlign': 'center'})
    else:
        genes_table = html.P("Gene association data columns missing.", style={'textAlign':'center'})

    kegg_df_cols = ['kegg_pathway_name', 'kegg_reaction_id', 'kegg_enzyme_id']
    if all(c in metabolite_associated_rows.columns for c in kegg_df_cols):
        kegg_df = metabolite_associated_rows[kegg_df_cols].drop_duplicates().copy()
        kegg_table = dash_table.DataTable(columns=[{"name": i.replace("_", " ").title(), "id": i} for i in kegg_df.columns], data=kegg_df.to_dict('records'), page_size=5, style_cell={'textAlign': 'left', 'padding': '5px'}, style_header={'fontWeight': 'bold'}) if not kegg_df.empty else html.P(f"No KEGG info for {selected_metabolite}.", style={'textAlign': 'center'})
    else:
        kegg_table = html.P("KEGG data columns missing.", style={'textAlign':'center'})
        
    tf_df_cols = ['transcription_factor', 'tf_activity_score']
    if all(c in metabolite_associated_rows.columns for c in tf_df_cols):
        tf_df = metabolite_associated_rows[tf_df_cols].drop_duplicates().copy()
        if not tf_df.empty:
            tf_df['tf_activity_score'] = tf_df['tf_activity_score'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        tf_table = dash_table.DataTable(columns=[{"name": i.replace("_", " ").title(), "id": i} for i in tf_df.columns], data=tf_df.to_dict('records'), page_size=5, style_cell={'textAlign': 'left', 'padding': '5px'}, style_header={'fontWeight': 'bold'}) if not tf_df.empty else html.P(f"No TFs associated with {selected_metabolite}.", style={'textAlign': 'center'})
    else:
        tf_table = html.P("TF data columns missing.", style={'textAlign':'center'})

    explanation_children = html.Div([
        html.H5(f"💡 Exploring: {selected_metabolite}", style=sticky_note_title_style),
        html.P(f"The cards display key identification, regulation stats, and chemical classification. Bar charts visualize specific chemical property groups. The 'Chemical Disposition Heuristics' section (adapted from Lipinski’s Rules) provides an indication of molecular properties relevant to bioavailability and cellular access. Further below, find a detailed properties table, a radar plot summarizing key attributes, and tables of associated genes, KEGG pathways/enzymes/reactions, and transcription factors from the dataset.")
    ])
    return (id_reg_content_children, classification_content_children, physchem_table_display, radar_fig_updated,
            bar_chart_size, bar_chart_polarity, bar_chart_lipophilicity, bar_chart_qm, lipinski_html,
            genes_table, kegg_table, tf_table, explanation_children)

# Gene Explorer Callbacks (Corrected structure)
@app.callback(
    [Output('gene-id-expression-card-explorer', 'children'),
     Output('associated-metabolites-table-gene-explorer', 'children'),
     Output('associated-kegg-info-table-gene-explorer', 'children'),
     Output('associated-tf-info-table-gene-explorer', 'children'),
     Output('gene-explorer-explanation', 'children')],
    [Input('gene-dropdown-explorer', 'value')]
)
def update_gene_details_explorer(selected_gene_symbol):
    no_data_card_msg = html.P("N/A", style={'textAlign': 'center', 'paddingTop':'20px'})
    no_data_table_msg = html.P("No data available.", style={'textAlign': 'center'})
    default_explanation = html.Div([html.H5("💡 Gene Explorer", style=sticky_note_title_style), html.P("Select a gene from the dropdown to view its expression details and associated biological entities like metabolites, pathways, and TFs.")])

    if not selected_gene_symbol or df.empty or 'gene_symbol' not in df.columns:
        return no_data_card_msg, no_data_table_msg, no_data_table_msg, no_data_table_msg, default_explanation

    gene_data_all_rows = df[df['gene_symbol'] == selected_gene_symbol]
    if gene_data_all_rows.empty:
        return no_data_card_msg, html.P(f"No metabolites associated with {selected_gene_symbol}."), html.P(f"No KEGG info for {selected_gene_symbol}."), html.P(f"No TFs associated with {selected_gene_symbol}."), default_explanation

    gene_data_unique_row = gene_data_all_rows.drop_duplicates(subset=['gene_symbol']).iloc[0]

    id_expression_content_children = [
        html.H4(f"Gene: {gene_data_unique_row.get('gene_symbol', 'N/A')}"),
        html.P(f"Log2 Fold Change: {gene_data_unique_row.get('gene_log2fc', float('nan')):.2f}" if pd.notna(gene_data_unique_row.get('gene_log2fc')) else "Log2FC: N/A"),
        html.P(f"Expression P-value: {gene_data_unique_row.get('gene_expr_pval', float('nan')):.2e}" if pd.notna(gene_data_unique_row.get('gene_expr_pval')) else "P-value: N/A"),
    ]
    
    met_cols = ['metabolite_name', 'metabolite_log2fc', 'metabolite_pval']
    if all(c in gene_data_all_rows.columns for c in met_cols):
        metabolites_df = gene_data_all_rows[met_cols].drop_duplicates().copy()
        if not metabolites_df.empty:
            metabolites_df['metabolite_log2fc'] = metabolites_df['metabolite_log2fc'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            metabolites_df['metabolite_pval'] = metabolites_df['metabolite_pval'].map(lambda x: f"{x:.2e}" if pd.notna(x) else "N/A")
        metabolites_table_display = dash_table.DataTable(columns=[{"name": "Metabolite", "id": "metabolite_name"}, {"name": "Log2FC", "id": "metabolite_log2fc"}, {"name": "P-value", "id": "metabolite_pval"}], data=metabolites_df.to_dict('records'), page_size=5, style_cell={'textAlign': 'left'}, style_header={'fontWeight': 'bold'}) if not metabolites_df.empty else html.P(f"No metabolites for {selected_gene_symbol}.")
    else:
        metabolites_table_display = html.P("Metabolite association data columns missing.")

    kegg_cols_g = ['kegg_pathway_name', 'kegg_reaction_id', 'kegg_enzyme_id']
    if all(c in gene_data_all_rows.columns for c in kegg_cols_g):
        kegg_df_gene = gene_data_all_rows[kegg_cols_g].drop_duplicates().copy()
        kegg_table_gene_display = dash_table.DataTable(columns=[{"name": i.replace("_"," ").title(), "id": i} for i in kegg_df_gene.columns], data=kegg_df_gene.to_dict('records'), page_size=5, style_cell={'textAlign': 'left'}, style_header={'fontWeight': 'bold'}) if not kegg_df_gene.empty else html.P(f"No KEGG info for {selected_gene_symbol}.")
    else:
        kegg_table_gene_display = html.P("KEGG data columns missing.")

    tf_cols_g = ['transcription_factor', 'tf_activity_score']
    if all(c in gene_data_all_rows.columns for c in tf_cols_g):
        tf_df_gene = gene_data_all_rows[tf_cols_g].drop_duplicates().copy()
        if not tf_df_gene.empty:
            tf_df_gene['tf_activity_score'] = tf_df_gene['tf_activity_score'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        tf_table_gene_display = dash_table.DataTable(columns=[{"name": "TF", "id": "transcription_factor"}, {"name": "Activity Score", "id": "tf_activity_score"}], data=tf_df_gene.to_dict('records'), page_size=5, style_cell={'textAlign': 'left'}, style_header={'fontWeight': 'bold'}) if not tf_df_gene.empty else html.P(f"No TFs for {selected_gene_symbol}.")
    else:
        tf_table_gene_display = html.P("TF data columns missing.")


    explanation_children = html.Div([
        html.H5(f"💡 Exploring: {selected_gene_symbol}", style=sticky_note_title_style),
        html.P("The card above shows the selected gene's expression summary. Tables below list associated metabolites, KEGG information (pathways, reactions, enzymes this gene is involved in), and any transcription factors associated with this gene in the dataset.")
    ])
    return id_expression_content_children, metabolites_table_display, kegg_table_gene_display, tf_table_gene_display, explanation_children


# Pathway Explorer Callbacks (Corrected structure)
@app.callback(
    [Output('pathway-dropdown-explorer', 'options'),
     Output('pathway-dropdown-explorer', 'value')],
    [Input('pathway-type-selector-explorer', 'value')]
)
def update_pathway_dropdown_options_explorer(selected_pathway_type):
    if selected_pathway_type == 'KEGG':
        options = [{'label': name, 'value': name} for name in unique_kegg_pathways_dropdown]
        default_value = unique_kegg_pathways_dropdown[0] if unique_kegg_pathways_dropdown else None
    elif selected_pathway_type == 'PROGENy':
        options = [{'label': name, 'value': name} for name in unique_progeny_pathways_dropdown]
        default_value = unique_progeny_pathways_dropdown[0] if unique_progeny_pathways_dropdown else None
    else: # Should not happen if value is constrained
        options, default_value = [], None
    return options, default_value

@app.callback(
    [Output('pathway-details-output-explorer', 'children'),
     Output('pathway-explorer-explanation', 'children')],
    [Input('pathway-dropdown-explorer', 'value')],
    [State('pathway-type-selector-explorer', 'value')]
)
def update_pathway_details_explorer(selected_pathway_name, pathway_type):
    default_explanation = html.Div([html.H5("💡 Pathway Explorer", style=sticky_note_title_style), html.P("Select a pathway type (KEGG or PROGENy) and then a specific pathway. Details and associated entities will appear here.")])
    error_message_component = html.Div([
        html.H5("⚠️ Error Loading Pathway Details", style={**sticky_note_title_style, 'color': 'red'}),
        html.P("An unexpected error occurred while trying to load the pathway details. Please try selecting a different pathway or contact support if the issue persists.")
    ], style={**sticky_note_style, 'borderLeft': '5px solid red'})

    try:
        # --- Enhanced Initial Input Validation ---
        # Log raw inputs immediately for debugging crashes that bypass later checks
        try:
            raw_selected_pathway_name_repr = repr(selected_pathway_name)
            raw_pathway_type_repr = repr(pathway_type)
            print(f"DEBUG update_pathway_details_explorer: Raw inputs - selected_pathway_name={raw_selected_pathway_name_repr}, pathway_type={raw_pathway_type_repr}")
        except Exception as log_e:
            print(f"CRITICAL DEBUG update_pathway_details_explorer: Error logging raw inputs: {log_e}")
            # If even logging inputs fails, it's a severe issue, return error component.
            return error_message_component, default_explanation

        # --- Explicit None checks and type validation at the very beginning ---
        if selected_pathway_name is None:
            print(f"Warning: selected_pathway_name is None in update_pathway_details_explorer.")
            return html.P("No pathway selected. Please select a pathway.", style={'textAlign': 'center', 'marginTop': '20px'}), default_explanation
        
        if not isinstance(selected_pathway_name, str) or not selected_pathway_name: # Check type and non-emptiness
            print(f"Warning: Invalid selected_pathway_name in update_pathway_details_explorer. Type: {type(selected_pathway_name)}, Value: '{str(selected_pathway_name)[:200]}' (truncated)")
            return html.P("Invalid or missing pathway name. Please select a valid pathway.", style={'textAlign': 'center', 'marginTop': '20px'}), default_explanation

        if pathway_type is None:
            print(f"Warning: pathway_type (from State) is None in update_pathway_details_explorer.")
            return html.P("Pathway type not available. Please select a pathway type first.", style={'textAlign': 'center', 'marginTop': '20px'}), default_explanation

        if not isinstance(pathway_type, str) or not pathway_type: # Check type and non-emptiness
            print(f"Warning: Invalid pathway_type (from State) in update_pathway_details_explorer. Type: {type(pathway_type)}, Value: '{str(pathway_type)[:200]}' (truncated)")
            return html.P("Invalid or missing pathway type. Please select a valid type.", style={'textAlign': 'center', 'marginTop': '20px'}), default_explanation

        if df.empty:
            print("Warning: Main DataFrame (df) is empty in update_pathway_details_explorer.")
            return html.P("Please select a pathway type and then a pathway.", style={'textAlign': 'center', 'marginTop': '20px'}), default_explanation

        details_content_children = [] # This will hold the H3, P, and tables
        explanation_children = default_explanation # Default explanation

        if pathway_type == 'KEGG': # Removed 'kegg_pathway_name' in df.columns check here, will be handled by pathway_data_rows.empty
            pathway_data_rows = df[df['kegg_pathway_name'] == selected_pathway_name]
            if pathway_data_rows.empty: return html.P(f"No data for KEGG pathway: {selected_pathway_name}."), default_explanation

            kegg_id_series = pathway_data_rows['kegg_pathway_id'].dropna().unique()
            kegg_id = kegg_id_series[0] if len(kegg_id_series) > 0 else "N/A"
            details_content_children.extend([html.H3(f"KEGG Pathway: {selected_pathway_name}"), html.P(f"KEGG ID: {kegg_id}"), html.Hr()])
            assoc_cols_map = {"Metabolites": ['metabolite_name', 'metabolite_log2fc', 'metabolite_pval'],
                              "Genes": ['gene_symbol', 'gene_log2fc', 'gene_expr_pval'],
                              "Enzymes": ['kegg_enzyme_id'], "Reactions": ['kegg_reaction_id']}
            explanation_children = html.Div([html.H5(f"💡 Exploring KEGG Pathway: {selected_pathway_name}", style=sticky_note_title_style), html.P("Details for the selected KEGG pathway, including associated metabolites, genes, enzymes, and reactions found in your dataset.")])

        elif pathway_type == 'PROGENy': # Removed 'progeny_pathway' in df.columns check here
            pathway_data_rows = df[df['progeny_pathway'] == selected_pathway_name]
            if pathway_data_rows.empty: return html.P(f"No data for PROGENy pathway: {selected_pathway_name}."), default_explanation

            progeny_unique_row_df = pathway_data_rows.drop_duplicates(subset=['progeny_pathway'])
            progeny_unique_row = progeny_unique_row_df.iloc[0] if not progeny_unique_row_df.empty else pd.Series()

            progeny_pval = progeny_unique_row.get('progeny_pathway_pval', float('nan'))
            progeny_score = progeny_unique_row.get('pathway_score', float('nan')) # Assuming 'pathway_score' is the PROGENy score
            details_content_children.extend([
                html.H3(f"PROGENy Pathway: {selected_pathway_name}"),
                html.P(f"Pathway Activity Score: {progeny_score:.2f}" if pd.notna(progeny_score) else "Activity Score: N/A"),
                html.P(f"Pathway P-value: {progeny_pval:.2e}" if pd.notna(progeny_pval) else "P-value: N/A"),
                html.Hr()
            ])
            assoc_cols_map = {"Associated TFs": ['transcription_factor', 'tf_activity_score'],
                              "Associated Genes": ['gene_symbol', 'gene_log2fc'],
                              "Associated Metabolites": ['metabolite_name', 'metabolite_log2fc']}
            explanation_children = html.Div([html.H5(f"💡 Exploring PROGENy Pathway: {selected_pathway_name}", style=sticky_note_title_style), html.P("Details for the selected PROGENy pathway, including its calculated activity score, p-value, and associated TFs, genes, and metabolites from your dataset.")])
        else:
            # This case should ideally be caught by the initial check or if pathway_type is unexpected.
            # However, if df structure is assumed (e.g. 'kegg_pathway_name' column must exist for KEGG),
            # then this is a valid fallback.
            return html.P(f"Invalid pathway type '{pathway_type}' or required columns not found in data."), default_explanation

        for entity_type_title, cols_list in assoc_cols_map.items():
            if not all(col in pathway_data_rows.columns for col in cols_list if col is not None): # Check if all primary ID cols are present
                details_content_children.append(html.P(f"Data columns for {entity_type_title} not fully available.", style={'textAlign': 'center'}))
                continue

            # Select only existing columns from cols_list
            existing_cols_in_df_for_entity = [col for col in cols_list if col in pathway_data_rows.columns]
            if not existing_cols_in_df_for_entity:
                details_content_children.append(html.P(f"No relevant data columns found for {entity_type_title}.", style={'textAlign': 'center'}))
                continue

            entity_df = pathway_data_rows[existing_cols_in_df_for_entity].drop_duplicates().copy()
            if entity_df.empty or entity_df.iloc[:,0].dropna().empty: # Check if primary entity column is empty
                details_content_children.append(html.P(f"No associated {entity_type_title.lower()} found for {selected_pathway_name}.", style={'textAlign': 'center'}))
            else:
                for col_format in ['metabolite_log2fc', 'gene_log2fc', 'tf_activity_score', 'pathway_score']:
                    if col_format in entity_df.columns: entity_df[col_format] = entity_df[col_format].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                for col_format_pval in ['metabolite_pval', 'gene_expr_pval', 'progeny_pathway_pval', 'tf_activity_pval']:
                    if col_format_pval in entity_df.columns: entity_df[col_format_pval] = entity_df[col_format_pval].map(lambda x: f"{x:.2e}" if pd.notna(x) else "N/A")

                table_cols = [{"name": c.replace("_", " ").title(), "id": c} for c in entity_df.columns]
                table = dash_table.DataTable(columns=table_cols, data=entity_df.to_dict('records'), page_size=5, style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace':'normal', 'height':'auto'}, style_header={'fontWeight': 'bold', 'backgroundColor': 'rgb(240, 240, 240)'}, sort_action="native")
                details_content_children.extend([html.H4(f"{entity_type_title}"), table, html.Br()])

        return html.Div(details_content_children), explanation_children

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()

        # Robustly try to get string representations for logging
        log_selected_pathway_name_repr = "Error getting repr(selected_pathway_name)."
        log_pathway_type_repr = "Error getting repr(pathway_type)." # Fixed duplicate variable name from previous suggestion
        error_repr = "Error getting repr(e)."

        try:
            log_selected_pathway_name_repr = repr(selected_pathway_name)
        except Exception:
            pass
        try:
            log_pathway_type_repr = repr(pathway_type)
        except Exception:
            pass
        try:
            error_repr = repr(e)
        except Exception:
            pass

        print(f"Error in update_pathway_details_explorer. Inputs: pathway_name={log_selected_pathway_name_repr}, pathway_type={log_pathway_type_repr}. Error: {error_repr}\nTraceback:\n{tb_str}")
        return error_message_component, default_explanation


# TF Explorer Callbacks (Corrected structure)
@app.callback(
    [Output('tf-id-activity-card-explorer', 'children'),
     Output('tf-differential-activity-plot-explorer', 'figure'),
     Output('associated-genes-table-tf-explorer', 'children'),
     Output('associated-metabolites-table-tf-explorer', 'children'),
     Output('associated-kegg-pathways-table-tf-explorer', 'children'),
     Output('tf-explorer-explanation', 'children')],
    [Input('tf-dropdown-explorer', 'value')]
)
def update_tf_details_explorer(selected_tf):
    no_data_card_msg = html.P("N/A", style={'textAlign': 'center', 'paddingTop':'20px'})
    no_data_table_msg = html.P("No data available.", style={'textAlign': 'center'})
    empty_fig = go.Figure(layout={'height': 300, 'annotations': [{'text': 'Select a TF to view its differential activity plot.', 'showarrow': False, 'font': {'size': 14}}]})
    default_explanation = html.Div([html.H5("💡 TF Explorer", style=sticky_note_title_style), html.P("Choose a Transcription Factor (TF) to see its activity scores, differential activity (if available), and lists of associated genes, metabolites, and pathways.")])

    if not selected_tf or df.empty or 'transcription_factor' not in df.columns:
        return no_data_card_msg, empty_fig, no_data_table_msg, no_data_table_msg, no_data_table_msg, default_explanation

    tf_data_all_rows = df[df['transcription_factor'] == selected_tf]
    if tf_data_all_rows.empty:
        return no_data_card_msg, empty_fig, html.P(f"No data for TF: {selected_tf}."), html.P(""), html.P(""), default_explanation

    tf_data_unique_row_df = tf_data_all_rows.drop_duplicates(subset=['transcription_factor'])
    tf_data_unique_row = tf_data_unique_row_df.iloc[0] if not tf_data_unique_row_df.empty else pd.Series()

    tf_activity_score = tf_data_unique_row.get('tf_activity_score', float('nan'))
    id_activity_content_children = [
        html.H4(f"TF: {tf_data_unique_row.get('transcription_factor', 'N/A')}"),
        html.P(f"Overall Activity Score: {tf_activity_score:.2f}" if pd.notna(tf_activity_score) else "Activity Score: N/A")
    ]

    diff_fig_updated = go.Figure(layout={'height': 300, 'title_text': "Differential Activity (Tumor vs Normal)"})
    mean_cols = ['tf_activity_tumor_mean', 'tf_activity_normal_mean']
    if all(col in tf_data_unique_row and pd.notna(tf_data_unique_row[col]) for col in mean_cols):
        tumor_mean = tf_data_unique_row.get('tf_activity_tumor_mean')
        normal_mean = tf_data_unique_row.get('tf_activity_normal_mean')
        pval = tf_data_unique_row.get('tf_activity_pval', float('nan'))
        diff_fig_updated.add_trace(go.Bar(name='Tumor Mean', x=['Activity'], y=[tumor_mean], marker_color='crimson'))
        diff_fig_updated.add_trace(go.Bar(name='Normal Mean', x=['Activity'], y=[normal_mean], marker_color='royalblue'))
        diff_fig_updated.update_layout(barmode='group', title_text=f"Differential Activity (p-val: {pval:.2e})" if pd.notna(pval) else "Differential Activity (p-val: N/A)", title_x=0.5, height=400, yaxis_title="Mean TF Activity")
    else:
        diff_fig_updated.update_layout(annotations=[{'text': 'Differential activity data not available.', 'showarrow': False, 'font': {'size': 14}}])

    genes_tf_cols = ['gene_symbol', 'gene_log2fc', 'gene_expr_pval']
    if all(c in tf_data_all_rows.columns for c in genes_tf_cols):
        genes_df_tf = tf_data_all_rows[genes_tf_cols].drop_duplicates().copy()
        if not genes_df_tf.empty:
            genes_df_tf['gene_log2fc'] = genes_df_tf['gene_log2fc'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            genes_df_tf['gene_expr_pval'] = genes_df_tf['gene_expr_pval'].map(lambda x: f"{x:.2e}" if pd.notna(x) else "N/A")
        genes_table_tf_display = dash_table.DataTable(columns=[{"name": "Gene", "id": "gene_symbol"}, {"name": "Log2FC", "id": "gene_log2fc"}, {"name": "P-value", "id": "gene_expr_pval"}], data=genes_df_tf.to_dict('records'), page_size=5, style_cell={'textAlign': 'left'}, style_header={'fontWeight': 'bold'}) if not genes_df_tf.empty else html.P(f"No genes associated with {selected_tf}.")
    else:
        genes_table_tf_display = html.P("Gene association data columns missing.")

    metabolites_tf_cols = ['metabolite_name', 'metabolite_log2fc', 'metabolite_pval']
    if all(c in tf_data_all_rows.columns for c in metabolites_tf_cols):
        metabolites_df_tf = tf_data_all_rows[metabolites_tf_cols].drop_duplicates().copy()
        if not metabolites_df_tf.empty:
            metabolites_df_tf['metabolite_log2fc'] = metabolites_df_tf['metabolite_log2fc'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            metabolites_df_tf['metabolite_pval'] = metabolites_df_tf['metabolite_pval'].map(lambda x: f"{x:.2e}" if pd.notna(x) else "N/A")
        metabolites_table_tf_display = dash_table.DataTable(columns=[{"name": "Metabolite", "id": "metabolite_name"}, {"name": "Log2FC", "id": "metabolite_log2fc"}, {"name": "P-value", "id": "metabolite_pval"}], data=metabolites_df_tf.to_dict('records'), page_size=5, style_cell={'textAlign': 'left'}, style_header={'fontWeight': 'bold'}) if not metabolites_df_tf.empty else html.P(f"No metabolites associated with {selected_tf}.")
    else:
        metabolites_table_tf_display = html.P("Metabolite association data columns missing.")

    kegg_path_tf_cols = ['kegg_pathway_name', 'kegg_pathway_id']
    if all(c in tf_data_all_rows.columns for c in kegg_path_tf_cols):
        kegg_pathways_df_tf = tf_data_all_rows[kegg_path_tf_cols].drop_duplicates().copy()
        kegg_pathways_table_tf_display = dash_table.DataTable(columns=[{"name": col.replace("_", " ").title(), "id": col} for col in kegg_pathways_df_tf.columns], data=kegg_pathways_df_tf.to_dict('records'), page_size=5, style_cell={'textAlign': 'left'}, style_header={'fontWeight': 'bold'}) if not kegg_pathways_df_tf.empty else html.P(f"No KEGG pathways associated with {selected_tf}.")
    else:
        kegg_pathways_table_tf_display = html.P("KEGG pathway association data columns missing.")

    explanation_children = html.Div([
        html.H5(f"💡 Exploring TF: {selected_tf}", style=sticky_note_title_style),
        html.P("The card presents the overall activity score for the selected TF. The bar chart visualizes its differential activity between conditions (e.g., tumor vs. normal mean activity) if this data is available. Tables below list associated genes (potential targets), metabolites whose levels might be influenced by or correlate with TF activity, and KEGG pathways where this TF or its targets might play a role.")
    ])
    return id_activity_content_children, diff_fig_updated, genes_table_tf_display, metabolites_table_tf_display, kegg_pathways_table_tf_display, explanation_children


# Network Explorer Callbacks
@app.callback(
    [Output('network-central-node-dropdown-explorer', 'options'),
     Output('network-central-node-dropdown-explorer', 'value')],
    [Input('network-node-type-selector-explorer', 'value')]
)
def update_central_node_dropdown_explorer(selected_type):
    if not selected_type or not TYPED_NODE_DROPDOWN_OPTIONS.get(selected_type):
        return [], None
    options = TYPED_NODE_DROPDOWN_OPTIONS[selected_type]
    value = options[0]['value'] if options else None
    return options, value

@app.callback(
    [Output('cytoscape-neighborhood-network-explorer', 'elements'),
     Output('network-explorer-explanation', 'children'),
     Output('tap-node-data-output-explorer', 'children', allow_duplicate=True)], # Allow duplicate and clear tap data
    [Input('network-central-node-dropdown-explorer', 'value')],
    prevent_initial_call=True # Avoids firing on load if dropdown has an initial value
)
def update_neighborhood_graph_explorer(selected_node_id):
    default_explanation = html.Div([html.H5("💡 Network Explorer", style=sticky_note_title_style), html.P("Select a node type (Metabolite, Gene, Pathway, or TF) and then a specific central node. This tool will then visualize the direct connections (neighborhood) of the selected node based on the relationships defined in your dataset. Click on nodes in the graph to see more details below.")])
    default_tap_output_message = "Network updated. Click a node to see its details and a contextual explanation here."

    if not selected_node_id:
        return [], default_explanation, default_tap_output_message
    
    central_node_obj_data = NODE_DETAILS_MAP.get(selected_node_id)
    if not central_node_obj_data:
        return [], html.Div([html.H5("⚠️ Node Not Found", style=sticky_note_title_style), html.P(f"Details for node ID '{selected_node_id}' could not be found.")], style=sticky_note_style), default_tap_output_message
    
    central_node_obj = {'data': central_node_obj_data}
    neighborhood_elements = [central_node_obj]
    neighbor_ids_in_graph = {selected_node_id} # Use a set to avoid duplicate nodes
    
    # Get direct edges efficiently using the map
    direct_edges = NODE_TO_EDGES_MAP.get(selected_node_id, [])
    neighborhood_elements.extend(direct_edges)

    # Collect neighbor IDs from the direct edges
    for edge in direct_edges:
        neighbor_ids_in_graph.add(edge['data']['source'])
        neighbor_ids_in_graph.add(edge['data']['target'])

    for node_id_in_graph in neighbor_ids_in_graph:
        if node_id_in_graph != selected_node_id: # Central node already added
            neighbor_node_data = NODE_DETAILS_MAP.get(node_id_in_graph)
            if neighbor_node_data:
                neighborhood_elements.append({'data': neighbor_node_data})
            else: # Add a placeholder if a neighbor ID from an edge isn't in NODE_DETAILS_MAP
                 neighborhood_elements.append({'data': {'id': node_id_in_graph, 'label': node_id_in_graph[:15]+'...', 'type': 'placeholder'}})


    node_label = central_node_obj_data.get('label', 'the selected node')
    explanation_children = html.Div([
        html.H5(f"💡 Network Neighborhood of: {node_label}", style=sticky_note_title_style),
        html.P(f"This network displays '{node_label}' (center) and its direct connections (1st degree neighbors) from the dataset. Node colors and shapes indicate their type (e.g., Metabolite - Red Ellipse, Gene - Blue Rectangle). Edges represent relationships. Click on any node to view its detailed information below the graph.")
    ])
    return neighborhood_elements, explanation_children, default_tap_output_message
    
@app.callback(
    Output('tap-node-data-output-explorer', 'children'),
    [Input('cytoscape-neighborhood-network-explorer', 'tapNodeData')]
)
def display_tap_node_data_explorer(data_tapped):
    default_message = "Select a central node above to build its neighborhood network. Then, click on any node in the graph to see its details and a contextual explanation here."
    error_message_component = html.Div([
        html.H5("⚠️ Error Displaying Node Data", style={**sticky_note_title_style, 'color': 'red'}),
        html.P("An unexpected error occurred while trying to display the tapped node's data.")
    ], style=sticky_note_style)

    try:
        if data_tapped:
            node_id = data_tapped.get('id')
            node_info_from_map = NODE_DETAILS_MAP.get(node_id, {}) # Use .get for safety
            node_label = node_info_from_map.get('label', 'Unknown Node')
            node_type = node_info_from_map.get('type', 'N/A')
            
            display_data_dict = node_info_from_map.get('full_data', data_tapped) 
            
            explanation_text = f"You've selected '{node_label}', which is a '{node_type}'. "
            if node_type == 'metabolite':
                explanation_text += "This small molecule is involved in metabolism. Check its connections to see related genes (enzymes), pathways, or other metabolites."
            elif node_type == 'gene':
                explanation_text += "This gene might encode an enzyme or regulatory protein. Its connections can reveal pathways it's part of or metabolites it influences/is influenced by."
            elif node_type == 'pathway':
                explanation_text += "This represents a biological pathway, a series of reactions. Explore its constituent genes and metabolites."
            elif node_type == 'tf':
                explanation_text += "This Transcription Factor can regulate the expression of connected genes, potentially influencing downstream metabolic processes."
            elif node_type == 'placeholder':
                 explanation_text += "This node is a placeholder, likely an ID from an edge that wasn't fully detailed in the node list. Its connections might still be informative."
            else: # Covers 'N/A' or any other unexpected type
                explanation_text += "Review its connections to understand its role in the displayed network."

            try:
                json_display_data = json.dumps(display_data_dict, indent=2)
            except TypeError: # Handle non-serializable content
                json_display_data = "Error: Tapped data contains non-serializable content."
            except Exception as json_e: # Catch any other json.dumps error
                print(f"Error serializing tapped node data for display: {json_e}")
                json_display_data = "Error: Could not serialize tapped data for display."

            details_children = [
                html.Strong(f"Selected: {node_label} (Type: {node_type})"),
                html.Hr(),
                html.P("Raw Data:", style={'fontWeight':'bold'}),
                html.Pre(json_display_data),
                html.Div([
                    html.H5("💡 About This Node", style=sticky_note_title_style), 
                    html.P(explanation_text)
                ], style=sticky_note_style)
            ]
            return html.Div(details_children)
        return default_message
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        log_data_tapped = str(data_tapped)[:500] if data_tapped is not None else "None" # Safe logging
        print(f"Error in display_tap_node_data_explorer. Input data_tapped (truncated): '{log_data_tapped}'. Error: {e}\nTraceback:\n{tb_str}")
        return error_message_component

# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8280)