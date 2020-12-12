/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;
using std::numeric_limits;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  default_random_engine gen;
  double std_x = std[0], std_y = std[1], std_theta = std[2];

  // For normal ditribution of x, y, and theta respectively
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  num_particles = 400;
  for (int i = 0; i < num_particles; i++) {
    particles.push_back(Particle{i, dist_x(gen), dist_y(gen), dist_theta(gen), 1});
    weights.push_back(1);
  }
  // Finish initialization
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
  double std_x = std_pos[0], std_y = std_pos[1], std_theta = std_pos[2];

  for(Particle& p: particles) {
    if (fabs(yaw_rate) > 0.001) {
  	  p.x += velocity / yaw_rate * (sin(p.theta + (yaw_rate * delta_t)) - sin(p.theta));
      p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + (yaw_rate * delta_t)));
      p.theta += yaw_rate * delta_t;
    } else {
      p.x += velocity * cos(p.theta) * delta_t;
      p.y += velocity * sin(p.theta) * delta_t;
    }
   
    // For normal ditribution of x, y, and theta respectively
    normal_distribution<double> dist_x(p.x, std_x);
    normal_distribution<double> dist_y(p.y, std_y);
    normal_distribution<double> dist_theta(p.theta, std_theta);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs obs = observations[i];
    // Placeholder for holding min distance between predicted and observed measurement
    double distance_min = numeric_limits<double>::infinity();

    // Find the most colosest predicted measurement
    for(LandmarkObs predicted_obs: predicted) {
      double current_distance = dist(obs.x, obs.y, predicted_obs.x, predicted_obs.y);
      if (distance_min < current_distance) {
        continue;
      }
      
      observations[i] = predicted_obs;
      // Set current distance as a distance min
      distance_min = current_distance;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double std_x = std_landmark[0], std_y = std_landmark[1];
  double gauss_norm = 1 / (2 * M_PI * std_x * std_y);

  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[i];
    double x_part = p.x;
    double y_part = p.y;
    double theta_part = p.theta;

    vector<LandmarkObs> transformed_observations;
    vector<LandmarkObs> associated_observations;
    vector<LandmarkObs> predicted_observations;
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    // Update each observation cordinates from vehicles to map system
    for (int i = 0; i < observations.size(); i++) {
      LandmarkObs obs = observations[i];
      double x_obs = obs.x;
      double y_obs = obs.y;

      // Get landmark on a map cordinate system
      LandmarkObs landmark = {i,
                              x_part + (cos(theta_part) * x_obs) - (sin(theta_part) * y_obs),
                              y_part + (sin(theta_part) * x_obs) + (cos(theta_part) * y_obs)};

      double distance = dist(x_part, y_part, landmark.x, landmark.y);
      if (distance > sensor_range) {
        continue;
      }

      // Set transformed observations
      transformed_observations.push_back(landmark);
      // Get associated observations
      associated_observations.push_back(landmark);
    }

    // Get predicted landmarks from Map
    for(Map::single_landmark_s landmark: map_landmarks.landmark_list) {
      double distance = dist(x_part, y_part, landmark.x_f, landmark.y_f);
      if (distance > sensor_range) {
        continue;
      }
      predicted_observations.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
    }

    // Update observations with nearest Map landmarks
    dataAssociation(predicted_observations, associated_observations);

    // Calculate weights
    for (int i = 0; i < associated_observations.size(); i++) {
      LandmarkObs transformed_obs = transformed_observations[i];
      LandmarkObs associated_obs = associated_observations[i];

      double exponent = (pow(transformed_obs.x - associated_obs.x, 2) / (2 * pow(std_x, 2)))
                      + (pow(transformed_obs.y - associated_obs.y, 2) / (2 * pow(std_y, 2)));

      // Update partial weight
      p.weight *= gauss_norm * exp(-exponent);

      // Update associations, sense_x, and sense_y
      associations.push_back(associated_obs.id);
      sense_x.push_back(associated_obs.x);
      sense_y.push_back(associated_obs.y);
    }

    // Update weights
    weights[i] = p.weight;

    // Set associations to partial
    SetAssociations(p, associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  discrete_distribution<int> disc(weights.begin(), weights.end());
  default_random_engine gen;
  vector<Particle> resampled_particles(num_particles);
  
  for (int i = 0; i < num_particles; i++) {
    resampled_particles[i] = particles[disc(gen)];
  }

  // Replace partiles with resampled one
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}