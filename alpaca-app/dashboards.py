from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import json

Base = declarative_base()

class UserDashboard(Base):
    __tablename__ = 'user_dashboards'
    id = Column(Integer, primary_key=True, autoincrement=True)
    dashboard_name = Column(String, nullable=False)
    elements = Column(JSON, nullable=False)
    layout = Column(JSON, nullable=False)

# Create a database connection
engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/alpaca") # db instead of localhost for docker

# Create tables (if they don't exist)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)

# Commit the table creation
session = Session()
session.commit()
session.close()

def save_or_update_dashboard(dashboard_name, elements, layout):
    elements_json = json.dumps(elements)
    layout_json = json.dumps(layout)
    
    session = Session()
    
    # Check if the dashboard with the given name exists for the user
    existing_dashboard = session.query(UserDashboard).filter_by(dashboard_name=dashboard_name).first()
    
    if existing_dashboard:
        # Update the existing dashboard configuration
        existing_dashboard.elements = elements_json
        existing_dashboard.layout = layout_json
    else:
        # Add a new dashboard entry
        new_dashboard = UserDashboard(dashboard_name=dashboard_name, elements=elements_json, layout=layout_json)
        session.add(new_dashboard)
    
    session.commit()
    session.close()

def dashboard_name_exists(dashboard_name):
    session = Session()
    
    # Check if a dashboard with the given name exists
    exists = session.query(UserDashboard).filter_by(dashboard_name=dashboard_name).first() is not None
    
    session.close()
    return exists

def get_dashboard_by_name(dashboard_name):
    session = Session()
    
    # Retrieve the dashboard by its name
    dashboard = session.query(UserDashboard).filter_by(dashboard_name=dashboard_name).first()
    
    session.close()
    
    if dashboard:
        elements = json.loads(dashboard.elements)
        layout = json.loads(dashboard.layout)
        return elements, layout
    else:
        return [], {}
    
def get_all_dashboard_names():
    session = Session()
    
    # Retrieve all dashboard names
    dashboard_names = session.query(UserDashboard.dashboard_name).all()
    
    session.close()
    
    return [name[0] for name in dashboard_names]