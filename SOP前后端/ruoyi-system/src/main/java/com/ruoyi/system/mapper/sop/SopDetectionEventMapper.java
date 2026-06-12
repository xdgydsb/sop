package com.ruoyi.system.mapper.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopDetectionEvent;

/**
 * SOP detection event mapper
 *
 * @author ruoyi
 */
public interface SopDetectionEventMapper
{
    public SopDetectionEvent selectSopDetectionEventByEventLogId(Long eventLogId);

    public List<SopDetectionEvent> selectSopDetectionEventList(SopDetectionEvent sopDetectionEvent);

    public int insertSopDetectionEvent(SopDetectionEvent sopDetectionEvent);

    public int updateSopDetectionEvent(SopDetectionEvent sopDetectionEvent);

    public int deleteSopDetectionEventByEventLogId(Long eventLogId);

    public int deleteSopDetectionEventByEventLogIds(Long[] eventLogIds);
}
