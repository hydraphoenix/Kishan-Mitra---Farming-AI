"""
Main Application Runner for AgroMRV System
Entry point for launching the agricultural MRV dashboard and system components
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import logging
from typing import Optional
import time
import signal

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config import AgroMRVConfig, setup_logging, validate_configuration

# Setup logging
setup_logging()
logger = logging.getLogger('agromrv')

class AgroMRVRunner:
    """Main application runner and manager"""
    
    def __init__(self):
        """Initialize the application runner"""
        self.streamlit_process = None
        self.dashboard_port = 8501
        self.dashboard_host = 'localhost'
        
        # Validate configuration on startup
        try:
            validate_configuration()
            logger.info("Configuration validation passed")
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        required_packages = [
            'streamlit',
            'pandas', 
            'numpy',
            'sklearn',  # scikit-learn imports as sklearn
            'plotly',
            'scipy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            print(f"Please install missing packages: pip install {' '.join(missing_packages)}")
            return False
        
        logger.info("All required dependencies are installed")
        return True
    
    def run_dashboard(self, host: str = None, port: int = None, debug: bool = False):
        """Launch the Streamlit dashboard"""
        
        if not self.check_dependencies():
            return False
        
        host = host or self.dashboard_host
        port = port or self.dashboard_port
        
        # Path to the dashboard app
        dashboard_path = PROJECT_ROOT / "app" / "dashboard" / "streamlit_app.py"
        
        if not dashboard_path.exists():
            logger.error(f"❌ Dashboard app not found: {dashboard_path}")
            return False
        
        # Streamlit command
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.address", host,
            "--server.port", str(port),
            "--server.headless", "true" if not debug else "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        # Add theme configuration
        cmd.extend([
            "--theme.base", "light",
            "--theme.primaryColor", "#2E8B57",
            "--theme.backgroundColor", "#FFFFFF",
            "--theme.secondaryBackgroundColor", "#F0F2F6"
        ])
        
        logger.info(f"Launching AgroMRV Dashboard...")
        logger.info(f"URL: http://{host}:{port}")
        logger.info(f"{AgroMRVConfig.APP_NAME} v{AgroMRVConfig.APP_VERSION}")
        
        try:
            self.streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Print startup information
            print(f"""
========================================
   {AgroMRVConfig.APP_NAME} v{AgroMRVConfig.APP_VERSION}
   {AgroMRVConfig.NABARD_HACKATHON}
========================================

Starting dashboard server...
URL: http://{host}:{port}
Target: {AgroMRVConfig.TARGET_AUDIENCE}
IPCC Tier 2 Compliant
AI/ML Powered
Blockchain Verified

Features Available:
   Real-time MRV data generation
   5 AI/ML prediction models  
   Blockchain verification system
   IPCC Tier 2 compliance
   Professional report export
   Interactive visualizations

Demo mode is enabled by default
Check the sidebar for navigation

Press Ctrl+C to stop the server
========================================
            """)
            
            # Monitor the process
            self._monitor_process()
            
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
            self.stop_dashboard()
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            return False
        
        return True
    
    def _monitor_process(self):
        """Monitor the Streamlit process"""
        try:
            while self.streamlit_process and self.streamlit_process.poll() is None:
                # Read output line by line
                line = self.streamlit_process.stdout.readline()
                if line:
                    # Filter out unnecessary Streamlit messages
                    if any(skip_msg in line for skip_msg in [
                        "You can now view your Streamlit app",
                        "Local URL:",
                        "Network URL:",
                        "For better performance"
                    ]):
                        continue
                    
                    # Log important messages
                    if "error" in line.lower() or "exception" in line.lower():
                        logger.error(line.strip())
                    elif "warning" in line.lower():
                        logger.warning(line.strip())
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Error monitoring process: {e}")
    
    def stop_dashboard(self):
        """Stop the dashboard server"""
        if self.streamlit_process:
            logger.info("Stopping dashboard server...")
            
            try:
                # Try graceful shutdown first
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                logger.warning("Forcing dashboard shutdown...")
                self.streamlit_process.kill()
                self.streamlit_process.wait()
            
            self.streamlit_process = None
            logger.info("Dashboard server stopped")
    
    def run_system_check(self):
        """Run comprehensive system check"""
        print(f"{AgroMRVConfig.APP_NAME} System Check")
        print("=" * 50)
        
        # Configuration check
        try:
            validate_configuration()
            print("[OK] Configuration: Valid")
        except Exception as e:
            print(f"[ERROR] Configuration: {e}")
            return False
        
        # Dependencies check
        if self.check_dependencies():
            print("[OK] Dependencies: All installed")
        else:
            print("[ERROR] Dependencies: Missing packages")
            return False
        
        # File structure check
        required_files = [
            "app/models/mrv_node.py",
            "app/models/ai_models.py", 
            "app/models/blockchain.py",
            "app/dashboard/streamlit_app.py",
            "app/utils/ipcc_compliance.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (PROJECT_ROOT / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"[ERROR] Files: Missing {missing_files}")
            return False
        else:
            print("[OK] Files: All core files present")
        
        # Feature flags check
        enabled_features = sum(AgroMRVConfig.FEATURE_FLAGS.values())
        total_features = len(AgroMRVConfig.FEATURE_FLAGS)
        print(f"[OK] Features: {enabled_features}/{total_features} enabled")
        
        # Test data generation
        try:
            sys.path.append(str(PROJECT_ROOT / "app"))
            from app.models.mrv_node import SmallholderMRVNode
            
            test_farm = SmallholderMRVNode("TEST001", "Punjab", "wheat", 2.5)
            test_data = test_farm.generate_historical_data(5)
            
            if len(test_data) == 5:
                print("[OK] Data Generation: Working")
            else:
                print("[ERROR] Data Generation: Failed")
                return False
                
        except Exception as e:
            print(f"[ERROR] Data Generation: {e}")
            return False
        
        # Test AI models
        try:
            from app.models.ai_models import AIModelManager
            
            ai_manager = AIModelManager()
            print("[OK] AI Models: Initialized")
            
        except Exception as e:
            print(f"[ERROR] AI Models: {e}")
            return False
        
        print("\nSystem check completed successfully!")
        print(f"Ready to run {AgroMRVConfig.APP_NAME}")
        
        return True
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("Installing AgroMRV dependencies...")
        
        try:
            # Check if requirements.txt exists
            requirements_file = PROJECT_ROOT / "requirements.txt"
            
            if requirements_file.exists():
                cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
                subprocess.run(cmd, check=True)
                print("[OK] Dependencies installed from requirements.txt")
            else:
                # Install core dependencies manually
                core_packages = [
                    "streamlit>=1.28.0",
                    "pandas>=1.5.0",
                    "numpy>=1.24.0", 
                    "scikit-learn>=1.3.0",
                    "plotly>=5.15.0",
                    "scipy>=1.11.0"
                ]
                
                for package in core_packages:
                    print(f"Installing {package}...")
                    cmd = [sys.executable, "-m", "pip", "install", package]
                    subprocess.run(cmd, check=True)
                
                print("[OK] Core dependencies installed")
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install dependencies: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Installation error: {e}")
            return False
        
        return True
    
    def create_desktop_shortcut(self):
        """Create desktop shortcut for easy access"""
        try:
            import os
            
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            shortcut_content = f"""@echo off
echo Starting AgroMRV System...
cd /d "{PROJECT_ROOT}"
python run_app.py dashboard
pause
"""
            
            shortcut_path = os.path.join(desktop_path, "AgroMRV_System.bat")
            
            with open(shortcut_path, 'w') as f:
                f.write(shortcut_content)
            
            print(f"[OK] Desktop shortcut created: {shortcut_path}")
            
        except Exception as e:
            print(f"[WARNING] Could not create desktop shortcut: {e}")

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\nShutdown signal received...")
    sys.exit(0)

def main():
    """Main entry point"""
    
    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description=f"{AgroMRVConfig.APP_NAME} v{AgroMRVConfig.APP_VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python run_app.py dashboard              # Launch dashboard (default)
  python run_app.py dashboard --port 8080  # Launch on custom port
  python run_app.py check                  # Run system check
  python run_app.py install                # Install dependencies
  
{AgroMRVConfig.NABARD_HACKATHON} - {AgroMRVConfig.NABARD_THEME}
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch Streamlit dashboard')
    dashboard_parser.add_argument('--host', default='localhost', help='Host address')
    dashboard_parser.add_argument('--port', type=int, default=8501, help='Port number')
    dashboard_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # System check command
    subparsers.add_parser('check', help='Run system check')
    
    # Install dependencies command
    subparsers.add_parser('install', help='Install required dependencies')
    
    # Create shortcut command
    subparsers.add_parser('shortcut', help='Create desktop shortcut')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = AgroMRVRunner()
    
    # Default to dashboard if no command specified
    if not args.command:
        args.command = 'dashboard'
        # Set default attributes for dashboard command when no subparser was used
        args.host = 'localhost'
        args.port = 8501
        args.debug = False
    
    try:
        if args.command == 'dashboard':
            success = runner.run_dashboard(
                host=getattr(args, 'host', 'localhost'),
                port=getattr(args, 'port', 8501),
                debug=getattr(args, 'debug', False)
            )
            sys.exit(0 if success else 1)
            
        elif args.command == 'check':
            success = runner.run_system_check()
            sys.exit(0 if success else 1)
            
        elif args.command == 'install':
            success = runner.install_dependencies()
            sys.exit(0 if success else 1)
            
        elif args.command == 'shortcut':
            runner.create_desktop_shortcut()
            sys.exit(0)
            
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        runner.stop_dashboard()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"[ERROR] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()