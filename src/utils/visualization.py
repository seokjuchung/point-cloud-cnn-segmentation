import numpy as np
import plotly.graph_objects as go

def visualize_point_cloud(points, labels=None, title='Point Cloud Visualization'):
    fig = go.Figure()

    if labels is not None:
        scatter = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=labels,
                colorscale='Viridis',
                opacity=0.8
            ),
            hoverinfo='text',
            text=[f'Energy: {point[3]} MeV' for point in points]
        )
    else:
        scatter = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=2, opacity=0.8)
        )

    fig.add_trace(scatter)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        width=800,
        height=600
    )
    fig.show()

def visualize_segmentation(points, predicted_labels, true_labels=None):
    fig = go.Figure()

    scatter_pred = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=predicted_labels,
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Predicted Labels'
    )
    fig.add_trace(scatter_pred)

    if true_labels is not None:
        scatter_true = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=true_labels,
                colorscale='Reds',
                opacity=0.5
            ),
            name='True Labels'
        )
        fig.add_trace(scatter_true)

    fig.update_layout(
        title='Segmentation Results',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        width=800,
        height=600
    )
    fig.show()