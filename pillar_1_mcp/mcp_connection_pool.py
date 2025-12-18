"""
MCP Connection Pool - Production-grade connection management
From: The Architect's Playbook, Pillar I
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)


class MCPConnectionPool:
    """Production-grade MCP connection pool with health monitoring"""
    
    def __init__(self, max_connections: int = 5, health_check_interval: int = 30):
        self.max_connections = max_connections
        self.health_check_interval = health_check_interval
        self.sessions: Dict[str, Any] = {}
        self.health_status: Dict[str, datetime] = {}
        self.tools_cache: Dict[str, list] = {}
        self.lock = asyncio.Lock()
    
    async def get_session(self, server_name: str, server_params: Any) -> Any:
        """Get or create a session for the specified server"""
        async with self.lock:
            if server_name in self.sessions:
                if await self._is_healthy(server_name):
                    return self.sessions[server_name]
                else:
                    await self._close_session(server_name)
            
            session = await self._create_session(server_name, server_params)
            self.sessions[server_name] = session
            self.health_status[server_name] = datetime.now()
            return session
    
    async def _create_session(self, server_name: str, server_params: Any) -> Any:
        """Create a new MCP session with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Import here to avoid issues if MCP not installed
                from mcp import ClientSession
                session = ClientSession(server_params)
                await session.initialize()
                logger.info(f"MCP session created for {server_name}")
                return session
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to create session for {server_name}: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise RuntimeError(f"Failed to create session for {server_name}")
    
    async def _is_healthy(self, server_name: str) -> bool:
        """Check if session is healthy via list_tools"""
        try:
            session = self.sessions[server_name]
            tools = await asyncio.wait_for(session.list_tools(), timeout=5.0)
            self.health_status[server_name] = datetime.now()
            self.tools_cache[server_name] = tools
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {server_name}: {e}")
            return False
    
    async def _close_session(self, server_name: str):
        """Safely close and remove a session"""
        if server_name in self.sessions:
            try:
                await self.sessions[server_name].close()
            except Exception as e:
                logger.error(f"Error closing session {server_name}: {e}")
            finally:
                del self.sessions[server_name]
                self.health_status.pop(server_name, None)
                self.tools_cache.pop(server_name, None)
    
    async def close_all(self):
        """Close all sessions gracefully"""
        for server_name in list(self.sessions.keys()):
            await self._close_session(server_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics for monitoring"""
        return {
            "active_sessions": len(self.sessions),
            "max_connections": self.max_connections,
            "sessions": {
                name: {
                    "last_health_check": self.health_status.get(name),
                    "tools_count": len(self.tools_cache.get(name, []))
                }
                for name in self.sessions
            }
        }
