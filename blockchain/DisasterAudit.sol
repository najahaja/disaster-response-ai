// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DisasterAudit {
    struct EventLog {
        uint256 timestamp;
        string agentId;
        string actionType;
        string location;
    }

    EventLog[] public auditLogs;

    event LogCreated(uint256 timestamp, string agentId, string actionType, string location);

    function logDisasterEvent(string memory _agentId, string memory _actionType, string memory _location) public {
        EventLog memory newLog = EventLog({
            timestamp: block.timestamp,
            agentId: _agentId,
            actionType: _actionType,
            location: _location
        });
        
        auditLogs.push(newLog);
        emit LogCreated(block.timestamp, _agentId, _actionType, _location);
    }

    function getLogsCount() public view returns (uint256) {
        return auditLogs.length;
    }

    function getLog(uint256 index) public view returns (uint256, string memory, string memory, string memory) {
        require(index < auditLogs.length, "Log index out of bounds");
        EventLog memory l = auditLogs[index];
        return (l.timestamp, l.agentId, l.actionType, l.location);
    }
}
