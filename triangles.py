# -------------
# IMPORTS
# -------------
from dash import Dash, html, dash_table, dcc
from dash import callback, Output, Input, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import io
import os
import ast
import base64
from zipfile import ZipFile
import json 

from utils.evaluator import SemanticSegmentationEvaluator
from utils.general import create_output_folders
#TODO: Allow column to be editted / deleted from table
#    : please allow experiment name to save csv to to prevent overwriting of classes when ur being lazy and running wo jfc noelle
#-----------------------------
# Globals
#-----------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Initialize the app
app = Dash(__name__, external_stylesheets=['/assets/style.css'], title="triangles")
# Initialize an empty DataFrame to store the classes and category IDs
df = pd.DataFrame(columns=["RGB", "Grayscale", "Class Name", "Category Name", "Category ID"])

#-----------------------------
# Create a heatmap figure for the confusion matrix
#-----------------------------
def create_confusion_matrix_figure_and_metrics(model_name, model_data):

    # Create a table of IoU and Dice per class
    table_columns = ["Class Name", "IoU", "Dice Coefficient"]
    table_data = [
        {
            "Class Name": class_name,
            "IoU": f"{IoU:.2f}",
            "Dice Coefficient": f"{Dice:.2f}"
        }
        for class_name, IoU, Dice in zip(model_data['class_names'], model_data['IoU_per_class'], model_data['dice_per_class'])
    ]

    # Add average IoU and Dice
    table_data.append({
        "Class Name": "Average",
        "IoU": f"{model_data['avg_IoU']:.2f}",
        "Dice Coefficient": f"{model_data['avg_dice']:.2f}"
    })

    # Create the confusion matrix heatmap
    conf_matrix_fig = go.Figure(data=go.Heatmap(
        z=model_data['confusion_matrix'],
        x=model_data['class_names'],
        y=model_data['class_names'],
        colorscale='electric'
    ))
    conf_matrix_fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
    )

    # Create the dash table component
    metrics_table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in table_columns],
        data=table_data,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
    )

    return metrics_table, dcc.Graph(figure=conf_matrix_fig, style={'width': '65%'})

#------------------------------
# App layout
#------------------------------
app.layout = [
    html.Div(className='row', children='TRIANGLES', style={'textAlign':'center','fontSize':30}),
    html.Div(className='row', children='A Semantic Segmentation Suite', style={'textAlign':'center','fontSize':24}),
    html.Hr(),
    html.B(className='row', children='Directions:', style={'textAlign':'left','fontSize':16}),
    html.Div(className='row', children='1. Upload a zip file containing the following subdirectories: (1) ground_truth and (2) 1+ model prediction output folders, in which each model output has its own subdirectory and the prediction image name maps to a corresponding ground truth image.', style={'textAlign':'left','fontSize':16}),
    html.Div(className='row', children='2. (Optional) Define classes and categories. If you do not input these, classes will automatically be determined for you and plots that rely on catrgories cannot be created :broken-heart:. If you include the void class 0 (grayscale) or (0,0,0) RGB, it must be the first label input.', style={'textAlign':'left','fontSize':16}),
    html.Div(className='row', children='3. Select the Run Evaluator button to visualize your results! This will take longer if step 2 was not taken. All metrics and plotly graphs will be saved to {ROOT_DIR}/{ZIP_FILE}/.', style={'textAlign':'left','fontSize':16}),
    html.Hr(),

    html.B("File Upload", style={'fontSize':16}),
    dcc.Upload(
        id='upload_prediction',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File'),
            ', (*.zip)'
        ]),
        style={
            'width': '98%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        accept=".zip",
        multiple=False,
    ),
    html.Div(id='output_uploaded', style={'textAlign':'center','fontSize':16}), # Placeholder for dynamic content
    html.Div(id='folder_path', style={'display': 'none'}),  # Hold the hidden folder path for evaluator
    # TODO: Allow them to name the experiment to create output files? Or just do exp={uploadfile}_{modelsubdir}?
    html.Div(html.B("Class and Category ID Input "), style={'fontSize':16,'marginBottom': '10px',}),# Option to upload a CSV or manually input form data
    dcc.Tabs(id="data-input-method", value='form', children=[
        dcc.Tab(label='Fill Out Form', value='form', children=[
            # Dropdown to select Grayscale or RGB
            html.Label("Select Label Type:"),
            dcc.Dropdown(
                id='label-type-dropdown',
                options=[
                    {'label': 'RGB', 'value': 'rgb'},
                    {'label': 'Grayscale', 'value': 'grayscale'}
                ],
                value='rgb',  # Default to RGB
                style={'marginBottom': '20px'}
            ),

            # Container for RGB inputs (initially shown)
            html.Div(id='rgb-input-container', children=[
                html.Label("Red (R):"),
                dcc.Input(id='input-r', type='number', min=0, max=255, placeholder="0-255", style={'marginRight': '10px'}),

                html.Label("Green (G):"),
                dcc.Input(id='input-g', type='number', min=0, max=255, placeholder="0-255", style={'marginRight': '10px'}),

                html.Label("Blue (B):"),
                dcc.Input(id='input-b', type='number', min=0, max=255, placeholder="0-255", style={'marginRight': '10px'}),
            ], style={'marginBottom': '20px'}),

            # Grayscale input (initially hidden)
            html.Div(id='grayscale-input-container', children=[
                html.Label("Grayscale Value:"),
                dcc.Input(id='input-grayscale', type='number', min=0, max=255, placeholder="0-255", style={'marginRight': '10px'}),
            ], style={'marginBottom': '20px', 'display': 'none'}),  # Initially hidden

            # Class Name Input (Optional)
            html.Div([
                html.Label("Class Name (Optional):"),
                dcc.Input(id='input-class-name', type='text', placeholder="Class Name", style={'marginRight': '10px'}),
            ], style={'marginBottom': '20px'}),

            # Category Name Input (Optional)
            html.Div([
                html.Label("Category Name (Optional):"),
                dcc.Input(id='input-category-name', type='text', placeholder="Category Name", style={'marginRight': '10px'}),
            ], style={'marginBottom': '20px'}),

            # Category ID Input
            html.Div([
                html.Label("Category ID:"),
                dcc.Input(id='input-category-id', type='number', placeholder="Category ID", style={'marginRight': '10px'}),
            ], style={'marginBottom': '20px'}),

            # Submit Button
            html.Button('Submit', id='submit-button', n_clicks=0, style={'marginBottom': '20px'})
        ]),

        dcc.Tab(label='Upload CSV', value='upload', children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a CSV File')
                ]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                    'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '20px',
                },
            )
        ]),
    ]),

    # Table to display the classes and category IDs
    dash_table.DataTable(
        id='class-table',
        columns=[
            {"name": "RGB", "id": "RGB", "editable": True},
            {"name": "Grayscale", "id": "Grayscale", "editable": True},
            {"name": "Class Name", "id": "Class Name", "editable": True},
            {"name": "Category Name", "id": "Category Name", "editable": True},
            {"name": "Category ID", "id": "Category ID", "editable": True},
        ],
        data=df.to_dict('records'),  # Empty initially
        style_table={'marginTop': '20px'}
    ),

    # Button to run the evaluator
    html.Button('Run Evaluator', id='run-evaluator', n_clicks=0, style={'marginTop': '20px'}),

     # Wrap the output in a loading spinner
    dcc.Loading(
        id="loading-spinner",
        type="circle",  # You can also use 'default' or 'cube'
        children = [
            html.H3("Model Performance Metrics", id='metrics-title',  style={'display': 'none'}),  # Hidden until data is uploaded
            # Output model eval result pretty but girl, so confusing
            html.Div(id='tabs-container'),
            dcc.Tabs(id='model-tabs', children=[], style={'display': 'none'}),  # Hidden until data is uploaded
            html.Div(
                id='side-by-side-container', 
                style={'display': 'flex', 'justifyContent': 'space-between'}  # Container for side-by-side layout
            )
        ]
    )
]


#------------------------------
# Upload Zip File
#------------------------------
@app.callback([Output('output_uploaded', 'children'),
                Output('folder_path', 'children')],
              [Input('upload_prediction', 'contents')],
              [State('upload_prediction', 'filename'),
               State('upload_prediction', 'last_modified')])
def update_output(contents, name, date):
    if contents is None:
        return 'NO FILE UPLOADED YET.', None
    else:
        # the content needs to be split. It contains the type and the real content
        content_type, content_string = contents.split(',')
        filename = name.split('.')[0]
        # Decode the base64 string
        content_decoded = base64.b64decode(content_string)
        # Use BytesIO to handle the decoded content
        zip_str = io.BytesIO(content_decoded)
        # Make the output folder
        output_folder = os.path.join(ROOT_DIR,'uploads')
        os.makedirs(output_folder, exist_ok=True)
        # Unzip file contents into respective folder 
        with ZipFile(zip_str, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        extracted_output_folder = os.path.join(output_folder,filename)
    return f'UPLOADED FILE: {name}', extracted_output_folder 


#------------------------------
# Callback to switch between RGB and Grayscale inputs
#------------------------------
@app.callback(
    [Output('rgb-input-container', 'style'),
     Output('grayscale-input-container', 'style')],
    Input('label-type-dropdown', 'value')
)
def toggle_input_fields(label_type):
    if label_type == 'rgb':
        # Show RGB inputs, hide grayscale
        return {'marginBottom': '20px'}, {'display': 'none'}
    else:
        # Show Grayscale input, hide RGB inputs
        return {'display': 'none'}, {'marginBottom': '20px'}


#------------------------------
# Callback to update the table when new values are submitted
#------------------------------
@app.callback(
    Output('class-table', 'data'),
    [Input('submit-button', 'n_clicks'), 
    Input('upload-data', 'contents')],
    [State('class-table', 'data'),
     State('label-type-dropdown', 'value'),
     State('input-r', 'value'),
     State('input-g', 'value'),
     State('input-b', 'value'),
     State('input-grayscale', 'value'),
     State('input-class-name', 'value'),
     State('input-category-name', 'value'),
     State('input-category-id', 'value'),
     State('upload-data', 'filename')]
)
def update_table(
    n_clicks, contents, current_data, label_type, 
    r_value, g_value, b_value, grayscale_value, 
    class_name, category_name, category_id, filename
):
    # If file uploaded, parse it
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df_uploaded = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df_uploaded.to_dict('records')

    # If form is submitted, add the new data
    if n_clicks > 0:
        if label_type == 'rgb' and r_value is not None and g_value is not None and b_value is not None and category_id is not None:
            current_data.append({
                "RGB": f"({r_value}, {g_value}, {b_value})",
                "Grayscale": '',
                "Class Name": class_name if class_name else '',
                "Category Name": category_name if category_name else '',
                "Category ID": category_id
            })
        elif label_type == 'grayscale' and grayscale_value is not None and category_id is not None:
            current_data.append({
                "RGB": '',
                "Grayscale": grayscale_value,
                "Class Name": class_name if class_name else '',
                "Category Name": category_name if category_name else '',
                "Category ID": category_id
            })

    return current_data

#------------------------------
# Callback to run the evaluator when the 'Run Evaluator' button is clicked
#------------------------------
@app.callback(
    [
        Output('metrics-title', 'style'),
        Output('model-tabs', 'children'),
        Output('model-tabs', 'style'),
        Output('side-by-side-container', 'children')
    ],
    Input('run-evaluator', 'n_clicks'),
    State('folder_path', 'children'),
    State('class-table', 'data'),
    State('label-type-dropdown', 'value')
)
def run_evaluator(n_clicks, folder_path, class_data, label_type):
    if n_clicks > 0 and folder_path:
        # Set up tabs
        tabs = []
        # Set up output dirs and save class data
        output_folder = create_output_folders(ROOT_DIR, folder_path, class_data)
        # Prepare the class data based on the label type
        classes = None
        #print(class_data)
        if label_type == 'rgb':
            mode = 'rgb'
            # Convert RGB strings back to tuples
            classes = [ast.literal_eval(row['RGB']) for row in class_data if row['RGB']]
            # Adding because confusion matrix calculations assumes void class label
            if (0,0,0) not in classes: classes.insert(0,(0,0,0)) 
        elif label_type == 'grayscale':
            mode = 'grayscale'
            # Use grayscale values directly
            classes = [row['Grayscale'] for row in class_data if row['Grayscale']]
            # Adding because confusion matrix calculations assumes void class label
            # And yes I do hate myself for this logic dyw
            if 0 not in classes: classes.insert(0,0) 
        # Set up class names 
        class_names = None
        if len(class_data) > 1 and class_data[0]['Class Name'] != '':
            class_names = [row['Class Name'] for row in class_data if row['Class Name']]
            # Removing void classes in a quite heinous way, but slightly less heinous than earlier attempts
            if 'unlabeled' in class_names: class_names.remove('unlabeled') 
            if 'void' in class_names: class_names.remove('void')
        # Initialize the evaluator with the folder path and class data
        evaluator = SemanticSegmentationEvaluator(directory=folder_path, classes=classes, class_names=class_names, mode=mode)

        # Run the evaluation
        results = evaluator.evaluate()

        # For each model in the returned dictionary, generate a metrics table and confusion matrix
        for model_name, model_data in results.items():
            metrics_table, confmat_fig = create_confusion_matrix_figure_and_metrics(model_name, model_data)
            tabs.append(dcc.Tab(label=model_name, value=model_name))
        # Save plotly figure and data
        pio.write_html(confmat_fig, file=os.path.join(output_folder,'visualizations','confmat.html'))
        with open(os.path.join(output_folder,'metrics','metrics.json'), 'w') as json_file:
            json.dump(results, json_file, indent=4)
        # Return tabs, display style, and metrics table+conf mat in a side by side format
        return {'display': 'block'}, tabs, {'display': 'block'}, [metrics_table, confmat_fig]  

    elif n_clicks > 0 and folder_path is None:
        # Return an empty figure if no button has been clicked
        # Cue pussycat dolls
        return {'display': 'none'}, [], {'display': 'none'}, None

    return {'display': 'none'}, [], {'display': 'none'}, None

#------------------------------
# Run app
#------------------------------
if __name__=='__main__':
    app.run(debug=True)

