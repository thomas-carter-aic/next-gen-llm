#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for Custom LLM
Document ID: monitoring_dashboard_20250705_080000
Created: July 5, 2025 08:00:00 UTC

This script creates a real-time monitoring dashboard for the deployed LLM system
using Streamlit for visualization and AWS CloudWatch for metrics collection.
"""

import streamlit as st
import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Nexus LLM Monitoring Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LLMMonitoringDashboard:
    def __init__(self, region='us-west-2'):
        """Initialize the monitoring dashboard."""
        self.region = region
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.ecs = boto3.client('ecs', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        
        # Cache for metrics to avoid excessive API calls
        if 'metrics_cache' not in st.session_state:
            st.session_state.metrics_cache = {}
            st.session_state.last_update = datetime.now()
    
    def get_cloudwatch_metrics(self, metric_name, namespace='NexusLLM/API', 
                              start_time=None, end_time=None, period=300):
        """Fetch metrics from CloudWatch."""
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.utcnow()
        
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=['Average', 'Sum', 'Maximum', 'Minimum']
            )
            
            # Convert to DataFrame
            data = []
            for point in response['Datapoints']:
                data.append({
                    'timestamp': point['Timestamp'],
                    'average': point.get('Average', 0),
                    'sum': point.get('Sum', 0),
                    'maximum': point.get('Maximum', 0),
                    'minimum': point.get('Minimum', 0)
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch CloudWatch metrics: {e}")
            return pd.DataFrame()
    
    def get_ecs_service_status(self, cluster_name='nexus-llm-cluster', service_name='nexus-llm-api'):
        """Get ECS service status."""
        try:
            response = self.ecs.describe_services(
                cluster=cluster_name,
                services=[service_name]
            )
            
            if response['services']:
                service = response['services'][0]
                return {
                    'service_name': service['serviceName'],
                    'status': service['status'],
                    'running_count': service['runningCount'],
                    'pending_count': service['pendingCount'],
                    'desired_count': service['desiredCount'],
                    'task_definition': service['taskDefinition'].split('/')[-1],
                    'created_at': service['createdAt'],
                    'updated_at': service.get('updatedAt', service['createdAt'])
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get ECS service status: {e}")
            return None
    
    def get_system_health_score(self):
        """Calculate overall system health score."""
        try:
            # Get recent metrics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=15)
            
            # Response time health (target: <2s)
            response_time_df = self.get_cloudwatch_metrics(
                'ResponseTime', start_time=start_time, end_time=end_time, period=60
            )
            response_time_health = 100
            if not response_time_df.empty:
                avg_response_time = response_time_df['average'].mean()
                if avg_response_time > 2:
                    response_time_health = max(0, 100 - (avg_response_time - 2) * 25)
            
            # Error rate health (target: <5%)
            error_df = self.get_cloudwatch_metrics(
                'ErrorCount', start_time=start_time, end_time=end_time, period=60
            )
            request_df = self.get_cloudwatch_metrics(
                'RequestCount', start_time=start_time, end_time=end_time, period=60
            )
            
            error_rate_health = 100
            if not error_df.empty and not request_df.empty:
                total_errors = error_df['sum'].sum()
                total_requests = request_df['sum'].sum()
                if total_requests > 0:
                    error_rate = (total_errors / total_requests) * 100
                    if error_rate > 5:
                        error_rate_health = max(0, 100 - (error_rate - 5) * 10)
            
            # Service availability health
            service_status = self.get_ecs_service_status()
            availability_health = 100
            if service_status:
                if service_status['running_count'] < service_status['desired_count']:
                    availability_health = (service_status['running_count'] / service_status['desired_count']) * 100
            else:
                availability_health = 0
            
            # Overall health score (weighted average)
            overall_health = (
                response_time_health * 0.4 +
                error_rate_health * 0.4 +
                availability_health * 0.2
            )
            
            return {
                'overall': round(overall_health, 1),
                'response_time': round(response_time_health, 1),
                'error_rate': round(error_rate_health, 1),
                'availability': round(availability_health, 1)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return {
                'overall': 0,
                'response_time': 0,
                'error_rate': 0,
                'availability': 0
            }
    
    def render_header(self):
        """Render dashboard header."""
        st.title("🤖 Nexus LLM Monitoring Dashboard")
        st.markdown("Real-time monitoring and analytics for your custom LLM deployment")
        
        # Auto-refresh toggle
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        with col2:
            auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
        with col3:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    def render_health_overview(self):
        """Render system health overview."""
        st.header("🏥 System Health Overview")
        
        health_scores = self.get_system_health_score()
        
        # Health score cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_health = health_scores['overall']
            color = "green" if overall_health >= 90 else "orange" if overall_health >= 70 else "red"
            st.metric(
                label="Overall Health",
                value=f"{overall_health}%",
                delta=None
            )
            st.markdown(f"<div style='color: {color}; font-weight: bold;'>{'🟢 Healthy' if overall_health >= 90 else '🟡 Warning' if overall_health >= 70 else '🔴 Critical'}</div>", 
                       unsafe_allow_html=True)
        
        with col2:
            response_health = health_scores['response_time']
            st.metric(
                label="Response Time",
                value=f"{response_health}%",
                delta=None
            )
        
        with col3:
            error_health = health_scores['error_rate']
            st.metric(
                label="Error Rate",
                value=f"{error_health}%",
                delta=None
            )
        
        with col4:
            availability_health = health_scores['availability']
            st.metric(
                label="Availability",
                value=f"{availability_health}%",
                delta=None
            )
    
    def render_service_status(self):
        """Render ECS service status."""
        st.header("🚀 Service Status")
        
        service_status = self.get_ecs_service_status()
        
        if service_status:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ECS Service Details")
                st.write(f"**Service Name:** {service_status['service_name']}")
                st.write(f"**Status:** {service_status['status']}")
                st.write(f"**Task Definition:** {service_status['task_definition']}")
                st.write(f"**Created:** {service_status['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                st.subheader("Task Status")
                
                # Task status pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=['Running', 'Pending'],
                    values=[service_status['running_count'], service_status['pending_count']],
                    hole=0.3,
                    marker_colors=['#00CC96', '#FFA15A']
                )])
                
                fig.update_layout(
                    title="Task Distribution",
                    height=300,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric(
                    label="Desired Tasks",
                    value=service_status['desired_count'],
                    delta=service_status['running_count'] - service_status['desired_count']
                )
        else:
            st.error("❌ Unable to fetch service status")
    
    def render_performance_metrics(self):
        """Render performance metrics charts."""
        st.header("📊 Performance Metrics")
        
        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            time_range = st.selectbox(
                "Time Range",
                ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
                index=1
            )
        
        with col2:
            metric_period = st.selectbox(
                "Granularity",
                ["1 minute", "5 minutes", "15 minutes", "1 hour"],
                index=1
            )
        
        # Convert selections to parameters
        time_ranges = {
            "Last Hour": timedelta(hours=1),
            "Last 6 Hours": timedelta(hours=6),
            "Last 24 Hours": timedelta(days=1),
            "Last 7 Days": timedelta(days=7)
        }
        
        periods = {
            "1 minute": 60,
            "5 minutes": 300,
            "15 minutes": 900,
            "1 hour": 3600
        }
        
        end_time = datetime.utcnow()
        start_time = end_time - time_ranges[time_range]
        period = periods[metric_period]
        
        # Fetch metrics
        response_time_df = self.get_cloudwatch_metrics('ResponseTime', start_time=start_time, end_time=end_time, period=period)
        request_count_df = self.get_cloudwatch_metrics('RequestCount', start_time=start_time, end_time=end_time, period=period)
        error_count_df = self.get_cloudwatch_metrics('ErrorCount', start_time=start_time, end_time=end_time, period=period)
        tokens_df = self.get_cloudwatch_metrics('TokensGenerated', start_time=start_time, end_time=end_time, period=period)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Time', 'Request Rate', 'Error Count', 'Tokens Generated'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Response Time
        if not response_time_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=response_time_df['timestamp'],
                    y=response_time_df['average'],
                    mode='lines+markers',
                    name='Avg Response Time',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # Request Rate
        if not request_count_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=request_count_df['timestamp'],
                    y=request_count_df['sum'],
                    mode='lines+markers',
                    name='Requests',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
        
        # Error Count
        if not error_count_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=error_count_df['timestamp'],
                    y=error_count_df['sum'],
                    mode='lines+markers',
                    name='Errors',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        
        # Tokens Generated
        if not tokens_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=tokens_df['timestamp'],
                    y=tokens_df['sum'],
                    mode='lines+markers',
                    name='Tokens',
                    line=dict(color='purple')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Performance Metrics Over Time"
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Seconds", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_cost_analysis(self):
        """Render cost analysis section."""
        st.header("💰 Cost Analysis")
        
        # Mock cost data (in production, this would come from AWS Cost Explorer API)
        current_month_cost = 287.45
        last_month_cost = 294.12
        projected_monthly_cost = 285.00
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current Month Cost",
                value=f"${current_month_cost:.2f}",
                delta=f"{current_month_cost - last_month_cost:.2f}"
            )
        
        with col2:
            st.metric(
                label="Projected Monthly Cost",
                value=f"${projected_monthly_cost:.2f}",
                delta=f"{projected_monthly_cost - current_month_cost:.2f}"
            )
        
        with col3:
            commercial_api_cost = 3300.00
            savings = commercial_api_cost - current_month_cost
            st.metric(
                label="Monthly Savings vs Commercial API",
                value=f"${savings:.2f}",
                delta=f"{(savings/commercial_api_cost)*100:.1f}% saved"
            )
        
        # Cost breakdown chart
        cost_breakdown = {
            'ECS Fargate': 167.82,
            'Application Load Balancer': 16.20,
            'S3 Storage': 4.61,
            'CloudWatch': 31.50,
            'Data Transfer': 9.00,
            'Other': 58.32
        }
        
        fig = px.pie(
            values=list(cost_breakdown.values()),
            names=list(cost_breakdown.keys()),
            title="Cost Breakdown by Service"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_and_logs(self):
        """Render alerts and recent logs."""
        st.header("🚨 Alerts & Logs")
        
        # Mock alerts data
        alerts = [
            {
                'timestamp': datetime.now() - timedelta(minutes=5),
                'severity': 'WARNING',
                'message': 'Response time exceeded 3 seconds for 2 consecutive minutes',
                'metric': 'ResponseTime'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'severity': 'INFO',
                'message': 'Auto-scaling event: Scaled up to 3 tasks due to high CPU utilization',
                'metric': 'AutoScaling'
            }
        ]
        
        if alerts:
            st.subheader("Recent Alerts")
            for alert in alerts:
                severity_color = {
                    'CRITICAL': '🔴',
                    'WARNING': '🟡',
                    'INFO': '🔵'
                }.get(alert['severity'], '⚪')
                
                st.write(f"{severity_color} **{alert['severity']}** - {alert['timestamp'].strftime('%H:%M:%S')}")
                st.write(f"   {alert['message']}")
                st.write("---")
        else:
            st.info("No recent alerts")
        
        # Recent logs (mock data)
        st.subheader("Recent Logs")
        logs = [
            "2025-07-05 08:15:23 - INFO - Request processed successfully in 1.2s",
            "2025-07-05 08:15:20 - INFO - Generated 156 tokens for user request",
            "2025-07-05 08:15:18 - INFO - Health check passed",
            "2025-07-05 08:15:15 - INFO - Request processed successfully in 0.8s"
        ]
        
        for log in logs:
            st.code(log)
    
    def render_sidebar(self):
        """Render sidebar with configuration options."""
        st.sidebar.header("⚙️ Configuration")
        
        # Region selector
        region = st.sidebar.selectbox(
            "AWS Region",
            ["us-west-2", "us-east-1", "eu-west-1"],
            index=0
        )
        
        # Cluster and service names
        cluster_name = st.sidebar.text_input(
            "ECS Cluster Name",
            value="nexus-llm-cluster"
        )
        
        service_name = st.sidebar.text_input(
            "ECS Service Name",
            value="nexus-llm-api"
        )
        
        st.sidebar.header("📋 Quick Actions")
        
        if st.sidebar.button("🔄 Restart Service"):
            st.sidebar.info("Service restart initiated...")
        
        if st.sidebar.button("📈 Scale Up"):
            st.sidebar.info("Scaling up service...")
        
        if st.sidebar.button("📉 Scale Down"):
            st.sidebar.info("Scaling down service...")
        
        st.sidebar.header("📊 Export Data")
        
        if st.sidebar.button("📥 Download Metrics"):
            st.sidebar.info("Preparing metrics export...")
        
        if st.sidebar.button("📋 Generate Report"):
            st.sidebar.info("Generating performance report...")
    
    def run_dashboard(self):
        """Run the complete dashboard."""
        self.render_header()
        self.render_sidebar()
        
        # Main content
        self.render_health_overview()
        st.divider()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            self.render_performance_metrics()
        with col2:
            self.render_service_status()
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            self.render_cost_analysis()
        with col2:
            self.render_alerts_and_logs()

def main():
    """Main function to run the dashboard."""
    dashboard = LLMMonitoringDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
