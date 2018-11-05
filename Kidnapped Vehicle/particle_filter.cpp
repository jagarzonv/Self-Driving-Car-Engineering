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

static default_random_engine rndm_eng;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).


	 num_particles = 50;//Set the number of particles
	 weights.resize(num_particles);

	  //http://www.cplusplus.com/reference/random/normal_distribution/
	  normal_distribution<double> Normdist_x(0, std[0]);
	  normal_distribution<double> Normdist_y(0, std[1]);
	  normal_distribution<double> Normdist_theta(0, std[2]);


	  for (int i = 0; i < num_particles; i++)//Initialize all particles to first position
	  {
	    Particle particle;
	    particle.id = i;
	    // Add random Gaussian noise to each particle
	   	//http://www.cplusplus.com/reference/random/
	    particle.x = x+Normdist_x(rndm_eng);
	    particle.y = y+Normdist_y(rndm_eng);
	    particle.theta = theta+Normdist_theta(rndm_eng);
	    particle.weight = 1.0;

	    particles.push_back(particle);
	  }

	  is_initialized = true;//flag

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    // references: https://github.com/jeremy-shannon/CarND-Kidnapped-Vehicle-Project

	  normal_distribution<double> Normdistnoise_x(0, std_pos[0]);
	  normal_distribution<double> Normdistnoise_y(0, std_pos[1]);
	  normal_distribution<double> Normdistnoise_theta(0, std_pos[2]);

	  for (int i = 0; i < num_particles; i++) { //Add measurements to each particle
		  //add random Gaussian noise

	    if (fabs(yaw_rate) < 0.0001)
	    {
	      particles[i].x += velocity * delta_t * cos(particles[i].theta)+ Normdistnoise_x(rndm_eng);
	      particles[i].y += velocity * delta_t * sin(particles[i].theta)+ Normdistnoise_y(rndm_eng);
	    }
	    else
	    {

	      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + Normdistnoise_x(rndm_eng);
	      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + Normdistnoise_y(rndm_eng);
	      particles[i].theta += yaw_rate * delta_t + Normdistnoise_theta(rndm_eng);
	    }


	  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
    // references: https://github.com/jeremy-shannon/CarND-Kidnapped-Vehicle-Project

	  for (unsigned int i = 0; i < observations.size(); i++)
	  {
	    LandmarkObs observation = observations[i];

	    double min_dist = numeric_limits<double>::max(); //double: 1.79769e+308 or 0x1.fffffffffffffp+1023
	    int id_map = -1;

	    for (unsigned int j = 0; j < predicted.size(); j++)
	    {
	      LandmarkObs pred = predicted[j];
	      double cur_dist = dist(observation.x, observation.y, pred.x, pred.y);
	      if (cur_dist < min_dist)
	      {
	        min_dist = cur_dist;
	        id_map = pred.id;
	      }
	    }
	    observations[i].id = id_map;
	  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    //   references: https://github.com/jessicayung/self-driving-car-nd/tree/master/term-2/p3-kidnapped-vehicle

	double x_2 = pow(std_landmark[0], 2);
	double y_2 = pow(std_landmark[1], 2);
	double xy_std = std_landmark[0] * std_landmark[1];

	for (int i=0; i < num_particles; i++) //Update the weights of each particle
	{
		Particle& particle = particles[i];
		// initialization
		long double mg_weight = 1;
		double p_x = particle.x;
		double p_y = particle.y;
		double p_theta = particle.theta;

		//observations from car coordinates to map coordinates
		vector<LandmarkObs> obs_transform;
		for (unsigned int j=0; j < observations.size(); j++)
		{
			LandmarkObs observation = observations[j];
			double o_x = observation.x;
			double o_y = observation.y;
			int   o_id = observation.id;
			double predicted_x = o_x * cos(p_theta) - o_y * sin(p_theta) + p_x;
			double predicted_y = o_x * sin(p_theta) + o_y * cos(p_theta) + p_y;
            obs_transform.push_back(LandmarkObs{o_id,predicted_x,predicted_y });

			Map::single_landmark_s near_landmark;
			double min_distance = sensor_range;
			double distance = 0;

			// associate sensor measurements to map landmarks
			vector<LandmarkObs> predictions;
			for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++) {
				Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
			    float lm_x = landmark.x_f;
			    float lm_y = landmark.y_f;
			    int   id_lm= landmark.id_i;
				//landmark-transformed distance
				distance = dist(predicted_x,predicted_y,lm_x,lm_y);
				// update nearest landmark to observation
				if (distance < min_distance) {
					min_distance = distance;
					near_landmark = landmark;
					predictions.push_back(LandmarkObs{id_lm, lm_x, lm_y });
				}
			}

			dataAssociation(predictions, obs_transform);

			//Multivariate Gaussian
			double mg_x = predicted_x - near_landmark.x_f;
			double mg_y = predicted_y - near_landmark.y_f;
			mg_weight *= exp(-0.5*((pow(mg_x,2))/x_2 + (pow(mg_y,2))/y_2))/(2*M_PI*xy_std);
		}
		particle.weight = mg_weight;
		weights[i] = mg_weight;

	}



}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    // references: https://github.com/jessicayung/self-driving-car-nd/tree/master/term-2/p3-kidnapped-vehicle

    discrete_distribution<> weights_(weights.begin(), weights.end());

    vector<Particle> new_particles;

    for (int i = 0; i < num_particles; ++i)
    {
        new_particles.push_back(particles[weights_(rndm_eng)]);
    }

    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
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

