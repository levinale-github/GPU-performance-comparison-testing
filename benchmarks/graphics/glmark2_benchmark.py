#!/usr/bin/env python3
"""
GPU Graphics Performance Benchmark

This script measures graphics performance using glmark2 on Linux
and a simple OpenGL test on other platforms. Results are reported
as a score indicating relative graphics performance.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import argparse
import logging
import platform
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Graphics-Benchmark")

# Add parent directory to path to import common utilities
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def detect_glmark2():
    """Check if glmark2 is available on the system"""
    return shutil.which("glmark2") is not None

def install_glmark2_instructions():
    """Provide platform-specific instructions for installing glmark2"""
    system = platform.system()
    if system == "Linux":
        distro = ""
        try:
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("ID="):
                        distro = line.split("=")[1].strip().strip('"')
                        break
        except:
            pass
        
        if distro == "ubuntu" or distro == "debian":
            return "Install glmark2 with: sudo apt-get install glmark2"
        elif distro == "fedora":
            return "Install glmark2 with: sudo dnf install glmark2"
        elif distro == "arch":
            return "Install glmark2 with: sudo pacman -S glmark2"
        else:
            return "Please install glmark2 using your distribution's package manager"
    else:
        return "glmark2 is primarily for Linux. This benchmark will use an alternative method for your platform."

def run_glmark2_benchmark():
    """Run the glmark2 benchmark and return the results"""
    try:
        result = subprocess.run(
            ["glmark2", "--fullscreen", "--annotate"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"glmark2 failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return None
        
        output = result.stdout
        logger.debug(output)
        
        # Extract the final score
        score = None
        for line in output.split('\n'):
            if "glmark2 Score:" in line:
                score = int(line.split(':')[1].strip())
                break
        
        if score is None:
            logger.error("Could not find glmark2 score in output")
            return None
        
        # Extract individual test results if available
        tests = {}
        for line in output.split('\n'):
            if "FPS:" in line and "(" in line and ")" in line:
                parts = line.split('(')[0].strip().split(':')
                if len(parts) >= 2:
                    test_name = parts[0].strip()
                    fps = float(parts[1].strip())
                    tests[test_name] = fps
        
        return {
            "GLMark2Score": score,
            "TestDetails": tests
        }
        
    except subprocess.TimeoutExpired:
        logger.error("glmark2 benchmark timed out after 5 minutes")
        return None
    except Exception as e:
        logger.error(f"Error running glmark2: {e}")
        return None

def try_import_graphics_libraries():
    """Try to import OpenGL and related libraries, returning available ones"""
    available_libs = {}
    
    try:
        import OpenGL
        from OpenGL import GL
        available_libs["opengl"] = True
    except ImportError:
        available_libs["opengl"] = False
        
    try:
        import pygame
        available_libs["pygame"] = True
    except ImportError:
        available_libs["pygame"] = False
        
    try:
        import PIL
        from PIL import Image
        available_libs["pillow"] = True
    except ImportError:
        available_libs["pillow"] = False
    
    try:
        import numpy as np
        available_libs["numpy"] = True
    except ImportError:
        available_libs["numpy"] = False
    
    return available_libs

def get_graphics_info():
    """Get information about the graphics capabilities using OpenGL"""
    try:
        import pygame
        from OpenGL import GL
        
        # Initialize pygame and create a window
        pygame.init()
        pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
        
        # Get OpenGL info
        gl_vendor = GL.glGetString(GL.GL_VENDOR).decode('utf-8')
        gl_renderer = GL.glGetString(GL.GL_RENDERER).decode('utf-8')
        gl_version = GL.glGetString(GL.GL_VERSION).decode('utf-8')
        
        # Clean up
        pygame.quit()
        
        return {
            "Vendor": gl_vendor,
            "Renderer": gl_renderer,
            "OpenGLVersion": gl_version
        }
    except Exception as e:
        logger.warning(f"Error getting OpenGL info: {e}")
        return {}

def run_opengl_simple_benchmark():
    """
    Run a simple OpenGL benchmark by rendering many triangles
    and measuring frame rate.
    """
    try:
        import pygame
        from OpenGL import GL
        import numpy as np
        import time
        
        # Initialize pygame and create a window
        pygame.init()
        screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
        
        # Simple vertex shader
        vertex_shader = """
        #version 330 core
        layout(location = 0) in vec3 vertexPosition;
        layout(location = 1) in vec3 vertexColor;
        out vec3 fragmentColor;
        uniform mat4 MVP;
        void main() {
            gl_Position = MVP * vec4(vertexPosition, 1);
            fragmentColor = vertexColor;
        }
        """
        
        # Simple fragment shader
        fragment_shader = """
        #version 330 core
        in vec3 fragmentColor;
        out vec3 color;
        void main() {
            color = fragmentColor;
        }
        """
        
        # Generate triangles
        num_triangles = 10000
        
        # Create a series of tests
        test_results = {}
        
        # Test with different numbers of triangles
        for test_triangles in [100, 1000, 10000, 50000]:
            # Run for 3 seconds per test
            frames = 0
            start_time = time.time()
            while time.time() - start_time < 3:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                
                # Clear the screen
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
                
                # Simple rendering... (simplified)
                # In a real benchmark, we would use shaders, buffers, etc.
                GL.glBegin(GL.GL_TRIANGLES)
                for i in range(test_triangles):
                    # Just draw a simple triangle
                    GL.glColor3f(1.0, 0.0, 0.0)
                    GL.glVertex3f(-0.5 + 0.1 * (i % 10), -0.5 + 0.1 * ((i // 10) % 10), 0.0)
                    GL.glColor3f(0.0, 1.0, 0.0)
                    GL.glVertex3f(0.0 + 0.1 * (i % 10), 0.5 + 0.1 * ((i // 10) % 10), 0.0)
                    GL.glColor3f(0.0, 0.0, 1.0)
                    GL.glVertex3f(0.5 + 0.1 * (i % 10), -0.5 + 0.1 * ((i // 10) % 10), 0.0)
                GL.glEnd()
                
                # Swap buffers
                pygame.display.flip()
                frames += 1
            
            elapsed = time.time() - start_time
            fps = frames / elapsed
            test_results[f"Triangles_{test_triangles}"] = fps
        
        # Clean up
        pygame.quit()
        
        # Calculate an overall score
        # Weight more triangles more heavily
        score = int(0.1 * test_results.get("Triangles_100", 0) +
                    0.2 * test_results.get("Triangles_1000", 0) +
                    0.3 * test_results.get("Triangles_10000", 0) +
                    0.4 * test_results.get("Triangles_50000", 0))
        
        return {
            "SimpleOpenGLScore": score,
            "TestDetails": test_results
        }
        
    except Exception as e:
        logger.error(f"Error running OpenGL benchmark: {e}")
        return None

def run_fallback_benchmark():
    """
    Run a very simple 2D benchmark for systems without OpenGL/pygame.
    This serves as a last resort to provide some basic performance data.
    """
    try:
        import time
        import numpy as np
        from PIL import Image, ImageDraw
        
        scores = {}
        
        # Create various sized images and measure performance
        for size in [256, 512, 1024, 2048]:
            # Create blank image
            img = Image.new('RGB', (size, size), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            start_time = time.time()
            frames = 0
            
            # Draw for 3 seconds
            while time.time() - start_time < 3:
                # Clear by creating a new drawing context
                draw = ImageDraw.Draw(img)
                
                # Draw random shapes
                for i in range(100):
                    x1 = np.random.randint(0, size)
                    y1 = np.random.randint(0, size)
                    x2 = np.random.randint(0, size)
                    y2 = np.random.randint(0, size)
                    
                    r = np.random.randint(0, 255)
                    g = np.random.randint(0, 255)
                    b = np.random.randint(0, 255)
                    
                    draw.line((x1, y1, x2, y2), fill=(r, g, b), width=1)
                
                frames += 1
            
            elapsed = time.time() - start_time
            fps = frames / elapsed
            scores[f"2D_Drawing_{size}x{size}"] = fps
        
        # Calculate weighted score
        score = int(0.1 * scores.get("2D_Drawing_256x256", 0) +
                    0.2 * scores.get("2D_Drawing_512x512", 0) +
                    0.3 * scores.get("2D_Drawing_1024x1024", 0) +
                    0.4 * scores.get("2D_Drawing_2048x2048", 0))
        
        return {
            "Simple2DGraphicsScore": score,
            "TestDetails": scores
        }
    except Exception as e:
        logger.error(f"Error running fallback benchmark: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="GPU Graphics Performance Benchmark")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()
    
    logger.info("GPU Graphics Performance Benchmark")
    logger.info("=" * 40)
    
    # Gather basic system info
    system = platform.system()
    logger.info(f"Running on {system} {platform.release()}")
    
    # Check if glmark2 is available on Linux
    has_glmark2 = False
    if system == "Linux":
        has_glmark2 = detect_glmark2()
        if not has_glmark2:
            logger.warning("glmark2 not found on your system.")
            logger.warning(install_glmark2_instructions())
    
    # Determine which benchmark to run
    graphics_results = None
    graphics_info = {}
    
    # Try to get graphics info regardless of which benchmark we'll run
    graphics_libs = try_import_graphics_libraries()
    if graphics_libs.get("opengl", False) and graphics_libs.get("pygame", False):
        graphics_info = get_graphics_info()
        logger.info(f"Detected graphics device: {graphics_info.get('Renderer', 'Unknown')}")
        logger.info(f"OpenGL version: {graphics_info.get('OpenGLVersion', 'Unknown')}")
    
    # Run the appropriate benchmark
    if system == "Linux" and has_glmark2:
        logger.info("Running glmark2 benchmark...")
        graphics_results = run_glmark2_benchmark()
        
        if graphics_results:
            logger.info(f"glmark2 score: {graphics_results['GLMark2Score']}")
        else:
            logger.warning("glmark2 benchmark failed. Trying OpenGL benchmark...")
            if graphics_libs.get("opengl", False) and graphics_libs.get("pygame", False):
                graphics_results = run_opengl_simple_benchmark()
    
    elif graphics_libs.get("opengl", False) and graphics_libs.get("pygame", False):
        logger.info("Running OpenGL benchmark...")
        graphics_results = run_opengl_simple_benchmark()
        
        if graphics_results:
            logger.info(f"OpenGL benchmark score: {graphics_results.get('SimpleOpenGLScore', 0)}")
        else:
            logger.warning("OpenGL benchmark failed.")
    
    # If all previous benchmarks failed or weren't applicable, try very simple 2D benchmark
    if not graphics_results and graphics_libs.get("pillow", False) and graphics_libs.get("numpy", False):
        logger.info("Running simple 2D graphics benchmark...")
        graphics_results = run_fallback_benchmark()
        
        if graphics_results:
            logger.info(f"2D Graphics score: {graphics_results.get('Simple2DGraphicsScore', 0)}")
    
    # If we couldn't run any benchmark
    if not graphics_results:
        logger.error("All graphics benchmarks failed.")
        logger.error("Please install the required dependencies:")
        logger.error("  pip install pygame numpy pillow PyOpenGL")
        if system == "Linux":
            logger.error("  - or - install glmark2 via your package manager")
        
        error_results = {
            "error": "All graphics benchmarks failed. Please install required dependencies."
        }
        
        if args.json:
            print(json.dumps(error_results, indent=2))
        else:
            print("\n" + json.dumps(error_results))
        
        return 1
    
    # Add graphics info to results
    if graphics_info:
        graphics_results["DeviceInfo"] = graphics_info
    
    # Normalize output - ensure we have a primary score called "GLMark2Score"
    # even if we ran a different benchmark
    if "GLMark2Score" not in graphics_results and "SimpleOpenGLScore" in graphics_results:
        graphics_results["GLMark2Score"] = graphics_results["SimpleOpenGLScore"]
        graphics_results["GLMark2Note"] = "Estimated from OpenGL test (not actual glmark2)"
    
    if "GLMark2Score" not in graphics_results and "Simple2DGraphicsScore" in graphics_results:
        graphics_results["GLMark2Score"] = graphics_results["Simple2DGraphicsScore"] // 2
        graphics_results["GLMark2Note"] = "Rough approximation from 2D test (not actual glmark2)"
    
    # Output results
    if args.json:
        # Only print the JSON result
        print(json.dumps(graphics_results, indent=2))
    else:
        # Print detailed results to console
        logger.info("\nSummary:")
        if "DeviceInfo" in graphics_results:
            logger.info(f"Device: {graphics_results['DeviceInfo'].get('Renderer', 'Unknown')}")
            logger.info(f"OpenGL: {graphics_results['DeviceInfo'].get('OpenGLVersion', 'Unknown')}")
        
        if "GLMark2Score" in graphics_results:
            logger.info(f"Graphics Score: {graphics_results['GLMark2Score']}")
            if "GLMark2Note" in graphics_results:
                logger.info(f"Note: {graphics_results['GLMark2Note']}")
        
        # Print JSON at the end for the runner to parse
        print("\n" + json.dumps(graphics_results))
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 