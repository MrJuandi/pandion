from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import inf
import unittest

from pandion.environment import (build_station_environment, delivered_weather,
                                  edge_arrival, equilibrium_moisture, validate_forcing)
from pandion.geographic_integrated import run_geographic_integrated


def fixture():
    origin=datetime(2022,6,17,10,6,tzinfo=timezone.utc)
    scenario={'activation_id':'EMSR578','seed':42,'provenance':{'image_acquired_at_utc':'2022-06-17T10:06:00Z'},
              'grid':{'rows':3,'cols':3,'cell_size_m':10},
              'fire':{'fuel_grid':[[101]*3 for _ in range(3)],'burn_mask':[[False]*3,[True,False,False],[False]*3],
                      'horizon_s':3600,'wind_to_m_s':[2,0],
                      'moisture':{'dead_1h':.04,'dead_10h':.05,'dead_100h':.07,'dead_herbaceous':.04,'live_herbaceous':.3,'live_woody':.6}},
              'observation_schedule':[]}
    variables={}
    for key,unit,value in [('air_temperature','°C',30),('relative_humidity','%',20),('precipitation','mm',0),('wind_speed_at_2m','m/s',2),('wind_direction_at_2m','°',270)]:
        records=[]
        for i in range(3):
            start=origin-timedelta(seconds=360)+timedelta(seconds=i*1800)
            records.append({'interval_start_utc':start.isoformat(),'interval_end_utc':(start+timedelta(seconds=1800)).isoformat(),'value':value,'usable':True,'quality_code':'V'})
        variables[key]={'unit':unit,'records':records}
    audit={'schema_version':'nightjar-pilot-weather-audit-0.1','station':{'code':'X6','name':'Baldomar'},'window':{'start_utc':origin.isoformat()},'variables':variables,'sources':{'provider':'Meteocat'}}
    return scenario,audit


class EnvironmentTests(unittest.TestCase):
    def test_simard_branches_units_and_relaxation(self):
        self.assertAlmostEqual(equilibrium_moisture(30,20),(2.22749+.160107*20-.014784*86)/100)
        self.assertGreater(equilibrium_moisture(30,80),equilibrium_moisture(30,5))
        s,a=fixture();e=build_station_environment(s,a)
        self.assertEqual(e['forcing'][0]['moisture']['dead_1h'],.04)
        self.assertNotEqual(e['forcing'][1]['moisture']['dead_1h'],.04)
        self.assertEqual(e['forcing'][1]['moisture']['live_woody'],.6)
        self.assertAlmostEqual(e['forcing'][0]['midflame_wind_to_m_s'][0],2)
        self.assertAlmostEqual(e['forcing'][0]['midflame_wind_to_m_s'][1],0,places=10)

    def test_completed_interval_delivery_and_clipped_horizon(self):
        s,a=fixture();e=build_station_environment(s,a);f=validate_forcing(e,3600)
        self.assertIsNone(delivered_weather(f,1439))
        self.assertEqual(delivered_weather(f,1440)['available_at_s'],1440)
        self.assertEqual(delivered_weather(f,3600)['available_at_s'],3240)
        self.assertEqual(f[-1]['end_s'],3600)
        self.assertEqual(f[-1]['available_at_s'],5040)
        report=delivered_weather(f,1440);report['moisture']['dead_1h']=999
        self.assertEqual(f[0]['moisture']['dead_1h'],.04)

    def test_invalid_sources_do_not_silently_drive_physics(self):
        s,a=fixture()
        for mutate in [lambda x:x['variables']['relative_humidity']['records'].pop(1),
                       lambda x:x['variables']['air_temperature'].update(unit='K'),
                       lambda x:x['variables']['wind_speed_at_2m']['records'][0].update(quality_code='N'),
                       lambda x:x['variables']['precipitation']['records'][0].update(value=1),
                       lambda x:x['variables']['air_temperature']['records'].append(deepcopy(x['variables']['air_temperature']['records'][0]))]:
            bad=deepcopy(a);mutate(bad)
            with self.assertRaises(ValueError):build_station_environment(s,bad)
        e=build_station_environment(s,a);e['forcing'][0]['available_at_s']=0
        with self.assertRaises(ValueError):validate_forcing(e,3600)

    def test_edge_integrates_weather_boundary(self):
        f=[{'start_s':0,'end_s':50},{'start_s':50,'end_s':100}]
        self.assertEqual(edge_arrival(0,100,f,lambda i:[1,2][i]),75)
        self.assertEqual(edge_arrival(40,100,f,lambda i:[1,2][i]),95)
        self.assertEqual(edge_arrival(0,201,f,lambda i:[1,2][i]),inf)
        self.assertEqual(edge_arrival(0,10,f,lambda i:[0,2][i]),55)

    def test_public_dem_reaches_controller_and_changes_fire(self):
        s,_=fixture();s['fire']['horizon_s']=600
        flat=run_geographic_integrated(s,'no_intervention')
        s['fire']['elevation_m']=[[c*8 for c in range(3)] for _ in range(3)]
        slope=run_geographic_integrated(s,'no_intervention')
        self.assertEqual(slope['decisions'][0]['observation']['elevation_m'],s['fire']['elevation_m'])
        self.assertNotEqual(flat['arrival_time_s'],slope['arrival_time_s'])

    def test_future_weather_cannot_change_past_observation(self):
        s,a=fixture();s['grid']['cell_size_m']=100;s['environment']=build_station_environment(s,a)
        first=run_geographic_integrated(s,'no_intervention',observation_mode='scheduled',_stop_after_steps=25)
        altered=deepcopy(s);altered['environment']['forcing'][1]['midflame_wind_to_m_s']=[-2,0]
        changed=run_geographic_integrated(altered,'no_intervention',observation_mode='scheduled',_stop_after_steps=25)
        self.assertEqual(first['decisions'],changed['decisions'])
        self.assertEqual(first['decisions'][0]['observation']['weather_status'],'not_yet_delivered')
        self.assertEqual(first['decisions'][0]['observation']['midflame_wind_to_m_s'],[0,0])
        self.assertIsNotNone(first['pending_observation']['environment_weather'])
        self.assertNotIn('forcing',first['pending_observation'])
        self.assertNotEqual(run_geographic_integrated(s,'no_intervention')['arrival_time_s'],
                            run_geographic_integrated(altered,'no_intervention')['arrival_time_s'])

    def test_realtime_weather_is_current_interval_only(self):
        s,a=fixture();s['grid']['cell_size_m']=100;s['environment']=build_station_environment(s,a)
        forcing=s['environment']['forcing']
        first=run_geographic_integrated(s,'no_intervention',_stop_after_steps=25)
        obs=first['decisions'][0]['observation']
        self.assertEqual(obs['observation_mode'],'realtime')
        self.assertEqual(obs['weather_status'],'realtime_station_interval')
        self.assertEqual(obs['midflame_wind_to_m_s'],forcing[0]['midflame_wind_to_m_s'])
        self.assertEqual(obs['fuel_moisture'],forcing[0]['moisture'])
        self.assertEqual(first['pending_observation']['environment_weather']['start_s'],forcing[1]['start_s'])
        # The last interval has not started at the stop time, so it cannot leak.
        altered=deepcopy(s);altered['environment']['forcing'][2]['midflame_wind_to_m_s']=[-2,0]
        changed=run_geographic_integrated(altered,'no_intervention',_stop_after_steps=25)
        self.assertEqual(first['decisions'],changed['decisions'])
        self.assertEqual(first['pending_observation'],changed['pending_observation'])

if __name__=='__main__':unittest.main()
