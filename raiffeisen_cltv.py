from dash import Dash, dcc, html, Input, Output, State
import dash_table
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import base64
import io

# Initialize the app
app = Dash(__name__)
server = app.server

# App Layout
app.layout = html.Div([
    html.H1("CLTV Prediction and EDA App", style={"textAlign": "center"}),

    # File upload section
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload'),

    html.Hr(),
    html.Div(id='eda-section'),

    html.Hr(),
    html.Div(id='cltv-section')
])


# Helper Function: Parse and Load Data
def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        return None
    return df


# Helper Function: Dynamically Detect Columns
def detect_columns(df):
    column_map = {
        "CustomerID": None,
        "InvoiceNo": None,
        "InvoiceDate": None,
        "Amount": None
    }
    for col in df.columns:
        if "customer" in col.lower():
            column_map["CustomerID"] = col
        elif "invoice" in col.lower():
            column_map["InvoiceNo"] = col
        elif "date" in col.lower():
            column_map["InvoiceDate"] = col
        elif "amount" in col.lower() or "total" in col.lower() or "quantity" in col.lower():
            column_map["Amount"] = col

    missing = [key for key, value in column_map.items() if value is None]
    if missing:
        return None, f"Missing required columns: {', '.join(missing)}"

    return column_map, None


# Helper Function: Calculate CLTV Features
def calculate_cltv_features(df, column_map):
    df['InvoiceDate'] = pd.to_datetime(df[column_map["InvoiceDate"]])

    # Snapshot date for Recency calculation
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Group by CustomerID
    cltv_df = df.groupby(column_map["CustomerID"]).agg({
        column_map["InvoiceDate"]: lambda x: (snapshot_date - x.max()).days,  # Recency
        column_map["InvoiceNo"]: 'count',  # Frequency
        column_map["Amount"]: 'sum'  # MonetaryValue
    }).reset_index()

    # Rename columns
    cltv_df.rename(columns={
        column_map["InvoiceDate"]: 'Recency',
        column_map["InvoiceNo"]: 'Frequency',
        column_map["Amount"]: 'MonetaryValue'
    }, inplace=True)

    return cltv_df


# Helper Function: CLTV Prediction
def predict_cltv(df):
    column_map, error = detect_columns(df)
    if error:
        return None, error

    cltv_df = calculate_cltv_features(df, column_map)

    # Synthetic CLTV target (example calculation)
    cltv_df['CLTV'] = cltv_df['MonetaryValue'] * cltv_df['Frequency'] / (cltv_df['Recency'] + 1)

    # Train-test split
    features = ['Recency', 'Frequency', 'MonetaryValue']
    target = 'CLTV'
    X = cltv_df[features]
    y = cltv_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict and calculate MSE
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return cltv_df, model, f"CLTV Prediction Model trained. MSE: {mse:.2f}"


# Helper Function: Generate Visualizations
def generate_visualizations(df):
    figs = []

    # Histogram
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        fig = px.histogram(df, x=col, title=f"Histogram of {col}")
        figs.append(fig)

    # Correlation Matrix
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    figs.append(fig)

    return figs


# Callbacks
@app.callback(
    [Output('output-data-upload', 'children'), Output('eda-section', 'children')],
    [Input('upload-data', 'contents'), State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents:
        df = parse_data(contents, filename)
        if df is None:
            return "Invalid file format", None

        # Perform EDA
        eda_html = [
            html.H3("EDA Report"),
            html.P(f"Shape: {df.shape}"),
            html.P(f"Missing Values: {df.isnull().sum().to_dict()}"),
            html.P(f"Duplicate Rows: {df.duplicated().sum()}")
        ]

        figs = generate_visualizations(df)
        figs_html = [dcc.Graph(figure=fig) for fig in figs]

        return f"Uploaded {filename}", eda_html + figs_html

    return None, None


@app.callback(
    Output('cltv-section', 'children'),
    [Input('upload-data', 'contents'), State('upload-data', 'filename')]
)
def update_cltv(contents, filename):
    if contents:
        df = parse_data(contents, filename)
        if df is None:
            return "Invalid file format"

        cltv_df, model, message = predict_cltv(df)
        if model:
            return html.Div([
                html.P(message),
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in cltv_df.columns],
                    data=cltv_df.to_dict('records'),
                    page_size=10
                )
            ])
        else:
            return html.Div([html.P(f"Error: {message}")])

    return None


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
