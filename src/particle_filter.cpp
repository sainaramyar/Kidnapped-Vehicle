/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	is_initialized = false;
	num_particles = 50;
	
	std::default_random_engine gen;
	
	
	
	std::normal_distribution<double> dist_x(x,std[0]);
	std::normal_distribution<double> dist_y(y,std[1]);
	std::normal_distribution<double> dist_theta(theta,std[2]);
	
	for (unsigned int i = 0; i < num_particles; ++i){
		struct Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		
		weights.push_back(1.0);
		particles.push_back(p);


		cout << x << y << theta;
	}
	
	is_initialized = true;


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	std::default_random_engine gen;
	// Creates a normal (Gaussian) distribution for x, y and yaw
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);
	
	for (int i=0;i<num_particles;i++){
		if (fabs(yaw_rate) > 1e-6){
			particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta+ yaw_rate*delta_t)) + dist_y(gen);
			particles[i].theta += yaw_rate*(delta_t) + dist_theta(gen);
		}
	else{
		particles[i].x += velocity *delta_t*cos(particles[i].theta) + dist_x(gen);
		particles[i].y += velocity *delta_t*sin(particles[i].theta) + dist_y(gen);
		particles[i].theta += dist_theta(gen);
		}
	}


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
			
		for (int i = 0; i < num_particles; i++) {

			std::vector<int> associations;
			std::vector<double> sense_x;
			std::vector<double> sense_y;


			double xp = particles[i].x;
			double yp = particles[i].y;
			double theta = particles[i].theta;
			double weight = 1;
			
			std::vector<LandmarkObs> observations_transf;
			for (LandmarkObs obsrv : observations){
				LandmarkObs obs_t;
				obs_t.id = obsrv.id;
				obs_t.x = xp + (cos(theta)*obsrv.x)-(sin(theta)*obsrv.y);
				obs_t.y = yp + (sin(theta)*obsrv.x)+(cos(theta)*obsrv.y);
				observations_transf.push_back(obs_t);
				} 

			for (LandmarkObs obs_t : observations_transf){
				vector <double> dists;
				vector <double> probs;
				
				for (Map::single_landmark_s landm : map_landmarks.landmark_list) {
					
                    double dst = dist(landm.x_f, landm.y_f, obs_t.x, obs_t.y);
					
                    dists.push_back(dst);
				}
				
				vector<double>::iterator result = min_element(begin(dists), end(dists));
				Map::single_landmark_s lm = map_landmarks.landmark_list[distance(begin(dists), result)];
				
				if (abs(lm.x_f - obs_t.x) < sensor_range && abs(lm.y_f - obs_t.y) < sensor_range){
				
				obs_t.id = lm.id_i;
				double p1 = (obs_t.x - lm.x_f)*(obs_t.x - lm.x_f)/(2*std_landmark[0]*std_landmark[0]);
				double p2 = (obs_t.y - lm.y_f)*(obs_t.y - lm.y_f)/(2*std_landmark[1]*std_landmark[1]);
				double prob = (1/(2*M_PI*std_landmark[0]*std_landmark[1]))*exp(-(p1 + p2));
				
				weight *= prob;

				sense_x.push_back(obs_t.x);
				sense_y.push_back(obs_t.y);
				associations.push_back(obs_t.id);
				
				}
			}
			particles[i].weight = weight;
			weights[i] = weight;
			particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
		}	

}

void ParticleFilter::resample() {

	default_random_engine gen;
    //std::mt19937 gen(rd());
    std::discrete_distribution<int> distribution(weights.begin(), weights.end());
    //std::map <int, int> m;
	std::vector<Particle> particles_new;
	
    for(int n=0; n<num_particles; n++) {
        particles_new.push_back(particles[distribution(gen)]);
    }
	
	particles = particles_new;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
	
	
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
