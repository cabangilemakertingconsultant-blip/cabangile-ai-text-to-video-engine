"""
environment.py

A production-grade environment verification module for Cabangile AI Studio v2.6.
Provides robust, cross-platform validation across Android (Termux), Linux, 
Windows, and macOS.

This module follows SOLID principles, Clean Architecture, and is optimized for
low-memory environments.
"""

import os
import sys
import shutil
import socket
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup logger
logger = logging.getLogger("CabangileAIStudio.Environment")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class EnvironmentReport:
    """Dataclass holding the structured environment verification report."""
    os_info: Dict[str, Any]
    python_version: str
    ffmpeg_status: Dict[str, Any]
    ffprobe_status: Dict[str, Any]
    internet_status: Dict[str, Any]
    disk_space: Dict[str, Any]
    memory_info: Dict[str, Any]
    api_providers: Dict[str, Any]
    font_status: Dict[str, Any]
    directory_status: Dict[str, Any]


class EnvironmentVerificationError(Exception):
    """Base exception for critical missing environment requirements."""
    pass


class EnvironmentVerifier:
    """
    Responsible solely for environment validation and report generation.
    Does not manipulate media assets. Optimized for low-memory constraints.
    """

    def __init__(
        self,
        required_directories: Optional[List[Path]] = None,
        min_disk_space_gb: float = 5.0,
        internet_timeout_sec: float = 3.0,
    ) -> None:
        self.required_directories = required_directories or [
            Path("output"),
            Path("cache"),
            Path("logs"),
        ]
        self.min_disk_space_gb = min_disk_space_gb
        self.internet_timeout_sec = internet_timeout_sec

    def detect_os(self) -> Dict[str, Any]:
        """Detects the operating system, explicitly isolating Android (Termux)."""
        logger.info("Detecting Operating System...")
        system_name = platform.system().lower()
        is_android = "android" in sys.platform or "ANDROID_DATA" in os.environ

        if is_android:
            platform_type = "Android (Termux)"
        elif system_name == "linux":
            platform_type = "Linux"
        elif system_name == "windows":
            platform_type = "Windows"
        elif system_name == "darwin":
            platform_type = "macOS"
        else:
            platform_type = f"Unknown ({platform.system()})"

        info = {
            "platform": platform_type,
            "release": platform.release(),
            "architecture": platform.machine(),
            "is_android": is_android
        }
        logger.info(f"OS Detected: {info['platform']} [{info['architecture']}]")
        return info

    def verify_python(self) -> str:
        """Verifies that the Python version is 3.11 or newer."""
        logger.info("Verifying Python version...")
        major, minor = sys.version_info.major, sys.version_info.minor
        version_str = f"{major}.{minor}.{sys.version_info.micro}"
        
        if (major, minor) < (3, 11):
            error_msg = f"Python 3.11+ required. Found version: {version_str}"
            logger.error(error_msg)
            raise EnvironmentVerificationError(error_msg)
            
        logger.info(f"Python version verified: {version_str}")
        return version_str

    def _verify_binary(self, binary_name: str) -> str:
        """Helper to locate and safely test a binary execution path."""
        path = shutil.which(binary_name)
        if not path:
            # Fallback pathing specifically for Termux environment structures
            termux_bin = f"/data/data/com.termux/files/usr/bin/{binary_name}"
            if os.path.exists(termux_bin):
                path = termux_bin

        if not path:
            error_msg = f"Required binary '{binary_name}' could not be located in PATH."
            logger.error(error_msg)
            raise EnvironmentVerificationError(error_msg)

        try:
            # Execute with minimal memory overhead to check valid performance
            subprocess.run(
                [path, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=5.0
            )
            logger.info(f"Binary '{binary_name}' successfully verified at: {path}")
            return path
        except (subprocess.SubprocessError, OSError) as e:
            error_msg = f"Binary '{binary_name}' found at {path} but failed to execute: {str(e)}"
            logger.error(error_msg)
            raise EnvironmentVerificationError(error_msg)

    def verify_ffmpeg(self) -> Dict[str, Any]:
        """Locates and validates both ffmpeg and ffprobe execution status."""
        logger.info("Verifying FFmpeg and FFprobe binaries...")
        ffmpeg_path = self._verify_binary("ffmpeg")
        ffprobe_path = self._verify_binary("ffprobe")
        return {
            "ffmpeg": {"status": "available", "path": ffmpeg_path},
            "ffprobe": {"status": "available", "path": ffprobe_path}
        }

    def verify_internet(self) -> Dict[str, Any]:
        """Checks internet connectivity with lightweight, low-overhead socket connection."""
        logger.info("Verifying Internet connectivity...")
        # Use Cloudflare's public DNS as a reliable, lightweight ping target
        host = "1.1.1.1"
        port = 53
        try:
            with socket.create_connection((host, port), timeout=self.internet_timeout_sec):
                logger.info("Internet connection established successfully.")
                return {"connected": True, "error": None}
        except OSError as e:
            logger.warning(f"Internet connection offline or timed out: {str(e)}")
            return {"connected": False, "error": str(e)}

    def check_disk_space(self) -> Dict[str, Any]:
        """Reports free space. Warns if below the configured fallback threshold."""
        logger.info("Checking available disk space...")
        # Path(".") handles systemic storage context safely across platforms
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024 ** 3)
        
        status = "healthy"
        if free_gb < self.min_disk_space_gb:
            status = "low_space_warning"
            logger.warning(
                f"Available disk space is critically low: {free_gb:.2f} GB available. "
                f"Target threshold: {self.min_disk_space_gb} GB."
            )
        else:
            logger.info(f"Disk space checks out: {free_gb:.2f} GB free.")

        return {
            "free_gb": round(free_gb, 2),
            "total_gb": round(total / (1024 ** 3), 2),
            "status": status
        }

    def check_memory(self) -> Dict[str, Any]:
        """Reports available RAM cleanly without forcing heavy external psutil dependencies."""
        logger.info("Checking available system memory...")
        system = platform.system().lower()
        is_android = "ANDROID_DATA" in os.environ
        available_mb: Optional[float] = None

        try:
            if is_android or system == "linux":
                # Efficient programmatic read of system info allocations
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                for line in meminfo.splitlines():
                    if "MemAvailable" in line:
                        available_mb = float(line.split()[1]) / 1024
                        break
                    elif "MemFree" in line and available_mb is None:
                        # Fallback for older Linux environments
                        available_mb = float(line.split()[1]) / 1024

            elif system == "windows":
                # Minimize execution weight by querying wmic parsing cleanly
                cmd = "wmic OS get FreePhysicalMemory /Value"
                out = subprocess.check_output(cmd, shell=True, text=True)
                for line in out.splitlines():
                    if "FreePhysicalMemory" in line:
                        available_mb = float(line.split("=")[1].strip()) / 1024
                        break

            elif system == "darwin":
                # Handle macOS dynamic page sizes seamlessly 
                vm_stat = subprocess.check_output(["vm_stat"], text=True)
                page_size = 4096  # Default fallback
                for line in vm_stat.splitlines():
                    if "page size of" in line:
                        page_size = int(line.split()[7])
                    if "Pages free" in line:
                        free_pages = int(line.split()[2].replace(".", ""))
                        available_mb = (free_pages * page_size) / (1024 * 1024)
                        break
        except Exception as e:
            logger.warning(f"Native memory check bypassed, platform non-standard: {str(e)}")

        if available_mb is not None:
            logger.info(f"Available Memory: {available_mb:.2f} MB")
            return {"available_mb": round(available_mb, 2), "status": "verified"}
        
        logger.warning("System memory visibility constrained. Parsing restricted.")
        return {"available_mb": "unknown", "status": "unsupported_or_restricted"}

    def verify_directories(self) -> Dict[str, Any]:
        """Verifies required paths exist. Creates missing targets automatically."""
        logger.info("Verifying working directory footprints...")
        status = {}
        for directory in self.required_directories:
            try:
                if not directory.exists():
                    directory.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created missing target directory: {directory}")
                    status[str(directory)] = "created"
                else:
                    status[str(directory)] = "verified"
            except Exception as e:
                logger.error(f"Failed to verify or create directory {directory}: {str(e)}")
                status[str(directory)] = f"failed: {str(e)}"
        return status

    def verify_apis(self) -> Dict[str, Any]:
        """Detects existence of cloud keys and tests local Ollama/Pollinations endpoints."""
        logger.info("Scanning AI Service Provider integrations...")
        
        # Check cloud API keys via environment variables
        openai_key = os.environ.get("OPENAI_API_KEY")
        stability_key = os.environ.get("STABILITY_API_KEY")

        report = {
            "openai": "available" if openai_key else "missing",
            "stability_ai": "available" if stability_key else "missing",
            "ollama": "missing",
            "pollinations": "missing"
        }

        # Check local Ollama availability via default endpoint ping
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        try:
            # Quick lightweight TCP socket connection instead of heavy HTTP requests
            host_parts = ollama_host.replace("http://", "").replace("https://", "").split(":")
            host = host_parts[0]
            port = int(host_parts[1]) if len(host_parts) > 1 else 11434
            with socket.create_connection((host, port), timeout=1.0):
                report["ollama"] = f"available ({ollama_host})"
        except OSError:
            report["ollama"] = "offline/unavailable"

        # Check Pollinations DNS routing reachability
        try:
            socket.gethostbyname("pollinations.ai")
            report["pollinations"] = "available"
        except OSError:
            report["pollinations"] = "unreachable"

        logger.info(f"API Check Complete -> OpenAI: {report['openai']}, Ollama: {report['ollama']}")
        return report

    def verify_fonts(self) -> Dict[str, Any]:
        """Verifies at least one usable system font asset is available across platforms."""
        logger.info("Scanning for compatible system TrueType fonts...")
        
        # Comprehensive cross-platform path targets for ttf files
        font_paths = [
            # Android / Termux standard fallback
            "/system/fonts/Roboto-Regular.ttf",
            "/system/fonts/DroidSans.ttf",
            # Linux common paths
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            # Windows standard path
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\segoeui.ttf",
            # macOS paths
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf"
        ]

        # Scan for alternative platform files if defaults miss
        scanned_dirs = [
            "/usr/share/fonts",
            "/system/fonts",
            "C:\\Windows\\Fonts",
            "/System/Library/Fonts"
        ]

        for path_str in font_paths:
            if os.path.exists(path_str):
                logger.info(f"Primary system font discovered: {path_str}")
                return {"status": "available", "primary_font": path_str}

        # Fallback shallow directory directory exploration for safety
        for directory in scanned_dirs:
            if os.path.exists(directory):
                for root, _, files in os.walk(directory):
                    for file in files:
                        if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                            found_font = os.path.join(root, file)
                            logger.info(f"Alternative font asset located: {found_font}")
                            return {"status": "available", "primary_font": found_font}
                    break # Shallow scan layer limit to protect processing overhead

        logger.warning("No functional system font infrastructure found. Rendering steps may fail.")
        return {"status": "missing", "primary_font": None}

    def generate_report(self) -> Dict[str, Any]:
        """
        Executes structural sequential validations to produce a comprehensive verification dictionary.
        Raises EnvironmentVerificationError if critical components fail execution checks.
        """
        logger.info("=== Starting Cabangile AI Studio v2.6 Environment Validation ===")
        
        os_info = self.detect_os()
        python_ver = self.verify_python()
        ffmpeg_info = self.verify_ffmpeg()
        internet_info = self.verify_internet()
        disk_info = self.check_disk_space()
        mem_info = self.check_memory()
        dir_info = self.verify_directories()
        api_info = self.verify_apis()
        font_info = self.verify_fonts()

        report = EnvironmentReport(
            os_info=os_info,
            python_version=python_ver,
            ffmpeg_status=ffmpeg_info["ffmpeg"],
            ffprobe_status=ffmpeg_info["ffprobe"],
            internet_status=internet_info,
            disk_space=disk_info,
            memory_info=mem_info,
            api_providers=api_info,
            font_status=font_info,
            directory_status=dir_info
        )

        logger.info("=== Environment Verification Check Complete [Status: STABLE] ===")
        
        # Return converted structural representation cleanly to matching consumer pipelines
        return {
            "operating_system": report.os_info,
            "python_version": report.python_version,
            "ffmpeg_status": report.ffmpeg_status,
            "ffprobe_status": report.ffprobe_status,
            "internet_status": report.internet_status,
            "disk_space": report.disk_space,
            "memory": report.memory_info,
            "api_provider_status": report.api_providers,
            "font_status": report.font_status,
            "directory_status": report.directory_status
        }
