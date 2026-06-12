package com.ruoyi.system.service.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopDetectionEvent;

/**
 * SOP detection event service
 *
 * @author ruoyi
 */
public interface ISopDetectionEventService
{
    public SopDetectionEvent selectSopDetectionEventByEventLogId(Long eventLogId);

    public List<SopDetectionEvent> selectSopDetectionEventList(SopDetectionEvent sopDetectionEvent);

    public SopDetectionEvent receiveVisualEvent(SopDetectionEvent sopDetectionEvent);

    public int insertSopDetectionEvent(SopDetectionEvent sopDetectionEvent);

    public int updateSopDetectionEvent(SopDetectionEvent sopDetectionEvent);

    public int deleteSopDetectionEventByEventLogIds(Long[] eventLogIds);

    public int deleteSopDetectionEventByEventLogId(Long eventLogId);
}
